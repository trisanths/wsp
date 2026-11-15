# Builds the fixed task slices used to test ablated models. Run once:
#   .venv/bin/python dissect/prepare_data.py
# Writes dissect/data/tasks.json. Everything is seeded, so re-running gives the same slices.
import json
import os
import random
import re


SEED = 0
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def clean_gsm8k_solution(text):
    # Drop calculator annotations like <<48/2=24>> and turn "#### 72" into a sentence.
    text = re.sub(r"<<[^>]*>>", "", text)
    parts = text.split("####")
    reasoning = parts[0].strip()
    answer = parts[1].strip().replace(",", "")
    return reasoning, answer


def load_dataset(*args, **kwargs):
    import datasets  # imported lazily so this file can be read without the library installed

    return datasets.load_dataset(*args, **kwargs)


def gsm8k_item(example):
    reasoning, answer = clean_gsm8k_solution(example["answer"])
    return {
        "question": example["question"].strip(),
        "solution": reasoning + "\nThe answer is " + answer + ".",
        "answer": answer,
    }


def wiki_chunks(split, how_many, rng):
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    paragraphs = []
    for row in data:
        line = row["text"].strip()
        if len(line) < 400:
            continue
        if line.startswith("="):
            continue
        paragraphs.append(line[:700])
    rng.shuffle(paragraphs)
    return paragraphs[:how_many]


def build_tasks():
    rng = random.Random(SEED)

    # GSM8K: test split -> 200 scoring + 100 held-out. Train split -> few-shot, proposal, mean reference.
    gsm = load_dataset("openai/gsm8k", "main")
    test_rows = list(gsm["test"])
    train_rows = list(gsm["train"])
    rng.shuffle(test_rows)
    rng.shuffle(train_rows)

    gsm8k_score = []
    for row in test_rows[0:200]:
        gsm8k_score.append(gsm8k_item(row))
    gsm8k_heldout = []
    for row in test_rows[200:300]:
        gsm8k_heldout.append(gsm8k_item(row))

    fewshot = []
    for row in train_rows[0:2]:
        fewshot.append(gsm8k_item(row))
    gsm8k_propose = []
    for row in train_rows[2:102]:
        gsm8k_propose.append(gsm8k_item(row))
    gsm8k_meanref = []
    for row in train_rows[102:134]:
        gsm8k_meanref.append(gsm8k_item(row))

    # TriviaQA (no context): factual recall control.
    trivia_stream = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
    trivia_pool = []
    for row in trivia_stream:
        trivia_pool.append({"question": row["question"].strip(), "answer": row["answer"]["value"].strip()})
        if len(trivia_pool) >= 600:
            break
    rng.shuffle(trivia_pool)
    trivia_fewshot = trivia_pool[0:3]
    trivia_score = trivia_pool[3:103]
    trivia_propose = trivia_pool[103:203]
    trivia_meanref = trivia_pool[203:235]

    # WikiText: plain-text control (all tokens) and a digits-in-prose control (numbers without arithmetic).
    wiki_score = wiki_chunks("test", 100, rng)
    wiki_train = wiki_chunks("train", 132, rng)
    wiki_propose = wiki_train[0:100]
    wiki_meanref = wiki_train[100:132]

    tasks = {
        "seed": SEED,
        "gsm8k_fewshot": fewshot,
        "gsm8k_score": gsm8k_score,
        "gsm8k_heldout": gsm8k_heldout,
        "gsm8k_propose": gsm8k_propose,
        "gsm8k_meanref": gsm8k_meanref,
        "trivia_fewshot": trivia_fewshot,
        "trivia_score": trivia_score,
        "trivia_propose": trivia_propose,
        "trivia_meanref": trivia_meanref,
        "wiki_score": wiki_score,
        "wiki_propose": wiki_propose,
        "wiki_meanref": wiki_meanref,
    }
    return tasks


def write_tasks(tasks):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "tasks.json")
    with open(path, "w") as f:
        json.dump(tasks, f, indent=1)
    for key in tasks:
        if isinstance(tasks[key], list):
            print(key, len(tasks[key]))
    print("wrote", path)


if __name__ == "__main__":
    write_tasks(build_tasks())
