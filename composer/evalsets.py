"""Practice and held-out sets, model scoring, and composer heal data. Runs on Modal.

Practice sets come from train (or validation) splits. Held-out sets are
test splits. The two never share items:
- gsm8k_practice is the fixed GSM8K search split (wsp.data); the
  gsm8k_train heal data is the rest of GSM8K train.
- math_practice, mbpp_practice, and medmcqa_practice are the first items
  of a seeded order of their train split; the math_train, mbpp_train, and
  medmcqa_train heal data use the items after them.
- mmlu_practice and law_practice come from MMLU validation and dev; the
  held-out MMLU sets come from MMLU test.

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

from composer.skills import LAW_SUBJECTS, extract_code, gsm8k_correct, last_boxed, math_correct

LETTERS = ["A", "B", "C", "D"]
MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                "prealgebra", "precalculus"]
SIZES = {"gsm8k_practice": 100, "math_practice": 100, "mbpp_practice": 150, "medmcqa_practice": 200,
         "mmlu_practice": 200, "law_practice": 200}
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


def medmcqa_train_rows():
    data = load("openlifescienceai/medmcqa", split="train")
    rows = []
    for index in seeded_order(len(data), 5):
        rows.append(data[index])
    return rows


# Generation sets.

def gsm8k_practice():
    from wsp.data import gsm8k_splits

    search, _ = gsm8k_splits(seed=0)
    items = []
    for row in search[:SIZES["gsm8k_practice"]]:
        items.append({"question": row["question"], "answer": row["answer"]})
    return items


def gsm8k_test():
    data = load("openai/gsm8k", "main", split="test")
    items = []
    for row in data:
        items.append({"question": row["question"], "answer": row["answer"]})
    return items


def math_train_rows():
    rows = []
    for config in MATH_CONFIGS:
        data = load("EleutherAI/hendrycks_math", config, split="train")
        for row in data:
            rows.append(row)
    order = seeded_order(len(rows), 3)
    out = []
    for index in order:
        out.append(rows[index])
    return out


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


def mbpp_train_order():
    rows = mbpp_rows("train")
    out = []
    for index in seeded_order(len(rows), 9):
        out.append(rows[index])
    return out


def humaneval():
    data = load("openai/openai_humaneval", split="test")
    items = []
    for row in data:
        items.append({"prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"]})
    return items


def load_set(name):
    """Return (kind, items) for a set name in composer.skills.SET_NOTES."""
    if name == "gsm8k_practice":
        return "gsm8k", gsm8k_practice()
    if name == "gsm8k_test":
        return "gsm8k", gsm8k_test()
    if name == "math_practice":
        return "math", math_items(math_train_rows()[:SIZES["math_practice"]])
    if name == "math500":
        return "math", math500()
    if name == "mbpp_practice":
        items = []
        for row in mbpp_train_order()[:SIZES["mbpp_practice"]]:
            items.append(mbpp_item(row))
        return "mbpp", items
    if name == "mbpp_test":
        items = []
        for row in mbpp_rows("test"):
            items.append(mbpp_item(row))
        return "mbpp", items
    if name == "humaneval":
        return "humaneval", humaneval()
    if name == "medmcqa_practice":
        items = []
        for row in medmcqa_train_rows()[:SIZES["medmcqa_practice"]]:
            items.append(medmcqa_item(row))
        return "mcq", items
    if name == "medmcqa_r10":
        return "mcq", medmcqa_r10()
    if name == "mmlu_practice":
        rows = mmlu_rows("validation")
        picked = []
        for index in seeded_order(len(rows), 11)[:SIZES["mmlu_practice"]]:
            picked.append(rows[index])
        return "mcq", mmlu_items(picked)
    if name == "law_practice":
        rows = mmlu_rows("validation", LAW_SUBJECTS) + mmlu_rows("dev", LAW_SUBJECTS)
        picked = []
        for index in seeded_order(len(rows), 13)[:SIZES["law_practice"]]:
            picked.append(rows[index])
        return "mcq", mmlu_items(picked)
    if name == "mmlu_dev":
        return "mcq", mmlu_dev()
    if name == "mmlu_law":
        return "mcq", mmlu_items(mmlu_rows("test", LAW_SUBJECTS))
    raise ValueError(f"unknown set {name!r}")


# Heal data for the composer (prompt, completion) pairs.

def math_train_pairs(tokenizer=None, seed=0, count=None, max_prompt_tokens=None):
    pairs = []
    for row in math_train_rows()[SIZES["math_practice"]:]:
        pairs.append(("Problem: " + row["problem"].strip() + "\nSolution:", " " + row["solution"].strip()))
        if count is not None and len(pairs) >= count:
            break
    return pairs


def mbpp_prompt(item):
    return ("You are an expert Python programmer. Write a Python function for this task: " + item["text"].strip() +
            "\nYour code should pass these tests:\n" + "\n".join(item["tests"]) +
            "\nReply with the code in a ```python code block.")


def mbpp_train_pairs(tokenizer=None, seed=0, count=None, max_prompt_tokens=None):
    rows = mbpp_train_order()[SIZES["mbpp_practice"]:] + mbpp_rows("validation") + mbpp_rows("prompt")
    pairs = []
    for row in rows:
        item = mbpp_item(row)
        pairs.append((mbpp_prompt(item) + "\n", "```python\n" + item["code"].strip() + "\n```"))
        if count is not None and len(pairs) >= count:
            break
    return pairs


def medmcqa_train_pairs(tokenizer=None, seed=0, count=8000, max_prompt_tokens=None):
    pairs = []
    for row in medmcqa_train_rows()[SIZES["medmcqa_practice"]:]:
        item = medmcqa_item(row)
        if item["gold"] < 0 or item["gold"] > 3:
            continue
        pairs.append((item["prompt"], " " + LETTERS[item["gold"]]))
        if count is not None and len(pairs) >= count:
            break
    return pairs


def mix_pairs(tokenizer=None, seed=0, count=8000, max_prompt_tokens=None):
    """An even mix of the math, code, medicine, and knowledge heal data."""
    from wsp.pairs import pairs_for

    each = 2000
    if count is not None:
        each = max(1, count // 5)
    # gsm8k_train uses seed 0: another seed moves the practice items into it.
    groups = [
        pairs_for("gsm8k_train", tokenizer, seed=0, count=each, max_prompt_tokens=None),
        math_train_pairs(count=each),
        mbpp_train_pairs(count=each),
        medmcqa_train_pairs(count=each),
        pairs_for("mmlu_aux", tokenizer, seed=seed, count=each),
    ]
    pairs = []
    for group in groups:
        pairs += list(group)
    random.Random(seed).shuffle(pairs)
    return pairs


def register_heal_data():
    """Add the composer heal data names to wsp.pairs."""
    from wsp.pairs import pair_names, register_pairs

    names = pair_names()
    if "math_train" not in names:
        register_pairs("math_train", math_train_pairs)
    if "mbpp_train" not in names:
        register_pairs("mbpp_train", mbpp_train_pairs)
    if "medmcqa_train" not in names:
        register_pairs("medmcqa_train", medmcqa_train_pairs, count=8000)
    if "composer_mix" not in names:
        register_pairs("composer_mix", mix_pairs, count=8000)


def heal_texts_for(name, tokenizer, seed=0):
    """Training texts for a heal data name: a wsp.pairs name or a wsp.data name.

    The data always comes from seed 0. The seed picks the GSM8K split
    (wsp.data.gsm8k_splits), and gsm8k_practice is the seed 0 search split,
    so any other seed would put practice items into the heal data. The
    program seed still sets the heal's own batch order (wsp.heal).
    """
    from wsp.pairs import pair_names, pairs_for

    if name in pair_names():
        texts = []
        for prompt, completion in pairs_for(name, tokenizer, seed=0):
            texts.append(prompt + completion)
        return texts
    from wsp.data import task_texts

    return task_texts(name, seed=0)


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
