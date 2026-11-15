# Builds the 5 built-in skill tests (baseline + control suite) once. Fixed seed.
#   .venv/bin/python mine/build_data.py      -> writes mine/data/builtin_skills.json
#
# Every item is {"context": str, "continuation": str, "digits_only": bool}.
# Scoring = mean log-prob of the continuation tokens given the context (teacher forced).
# Each skill has a "search" half (the agent may see results on it) and a "heldout" half
# (only used inside the gate; the agent only ever sees pass/fail for it).
import json
import os
import random
import re

SEED = 0
N_SEARCH = 100
N_HELDOUT = 50
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "builtin_skills.json")


def load(*args, **kwargs):
    import datasets
    return datasets.load_dataset(*args, **kwargs)


NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def final_number(text):
    found = NUMBER.findall(text)
    if not found:
        return None
    return found[-1].replace(",", "").rstrip(".")


def orca_solution(text):
    # Orca-Math answers are worked solutions; end each with the answer line that confirm_math parses
    answer = final_number(text)
    return text.strip() + "\nThe answer is " + answer + "."


def build_math(rng):
    # Approved pool from wsp/healdata.py (no benchmark data in any split): Orca-Math word problems.
    stream = load("microsoft/orca-math-word-problems-200k", split="train", streaming=True)
    rows = []
    for row in stream:
        q = row["question"].strip()
        a = row["answer"].strip()
        if final_number(a) is None or len(a) > 500 or len(q) > 400:
            continue
        rows.append({"question": q, "answer": a})
        if len(rows) >= 4000:
            break
    rng.shuffle(rows)
    shots = ""
    for row in rows[:2]:
        shots = shots + "Question: " + row["question"] + "\nAnswer: " + orca_solution(row["answer"]) + "\n\n"
    items = []
    for row in rows[2:2 + N_SEARCH + N_HELDOUT]:
        items.append({
            "context": shots + "Question: " + row["question"] + "\nAnswer:",
            "continuation": " " + orca_solution(row["answer"]),
            "digits_only": True,  # score only digit tokens: the arithmetic, not the wording
            "answer": final_number(row["answer"]),
            "question": row["question"],
        })
    return items


def looks_like_python(code):
    return ("def " in code or "print(" in code or "import " in code or "return " in code) and "{" not in code[:1]


def build_code(rng):
    # Approved pool from wsp/healdata.py (no benchmark data in any split): CodeAlpaca Python answers.
    data = load("sahil2801/CodeAlpaca-20k", split="train")
    rows = []
    for row in data:
        code = (row.get("output") or "").strip().replace("\r\n", "\n")
        task = row["instruction"].strip()
        if row.get("input"):
            task = task + " Input: " + row["input"].strip()
        if not looks_like_python(code) or len(code) > 600 or len(code) < 40 or "\n" in task:
            continue
        rows.append({"task": task, "code": code})
    rng.shuffle(rows)
    shot = rows[0]
    shots = "# Task: " + shot["task"] + "\n" + shot["code"] + "\n\n"
    items = []
    for row in rows[1:1 + N_SEARCH + N_HELDOUT]:
        items.append({"context": shots + "# Task: " + row["task"] + "\n", "continuation": row["code"],
                      "digits_only": False})
    return items


def build_facts(rng):
    stream = load("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
    rows = []
    for row in stream:
        answer = row["answer"]["value"].strip()
        if 0 < len(answer) <= 30:
            rows.append({"q": row["question"].strip(), "a": answer})
        if len(rows) >= 600:
            break
    rng.shuffle(rows)
    shots = ""
    for r in rows[:3]:
        shots = shots + "Q: " + r["q"] + "\nA: " + r["a"] + "\n\n"
    items = []
    for r in rows[3:3 + N_SEARCH + N_HELDOUT]:
        items.append({"context": shots + "Q: " + r["q"] + "\nA:", "continuation": " " + r["a"], "digits_only": False})
    return items


def build_multilingual(rng):
    # English -> German / French / Spanish / Italian / Portuguese translation (OPUS-100 test sets)
    pairs = [("de", "German"), ("fr", "French"), ("es", "Spanish"), ("it", "Italian"), ("pt", "Portuguese")]
    per_lang = (N_SEARCH + N_HELDOUT) // len(pairs)
    items = []
    for code, name in pairs:
        config = "de-en" if code == "de" else "en-" + code
        data = list(load("Helsinki-NLP/opus-100", config, split="test"))
        good = []
        for row in data:
            en = row["translation"]["en"].strip()
            other = row["translation"][code].strip()
            if 40 <= len(en) <= 160 and 20 <= len(other) <= 200:
                good.append((en, other))
        rng.shuffle(good)
        shots = ""
        for en, other in good[:2]:
            shots = shots + "English: " + en + "\n" + name + ": " + other + "\n\n"
        for en, other in good[2:2 + per_lang]:
            items.append({"context": shots + "English: " + en + "\n" + name + ":", "continuation": " " + other,
                          "digits_only": False})
    rng.shuffle(items)
    return items


def build_general_text(rng):
    data = load("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    paragraphs = []
    for row in data:
        line = row["text"].strip()
        if len(line) >= 600 and not line.startswith("="):
            paragraphs.append(line)
    rng.shuffle(paragraphs)
    items = []
    for p in paragraphs[:N_SEARCH + N_HELDOUT]:
        items.append({"context": p[:200], "continuation": p[200:600], "digits_only": False})
    return items


def split(items):
    return {"search": items[:N_SEARCH], "heldout": items[N_SEARCH:N_SEARCH + N_HELDOUT]}


def main():
    builders = [("math", build_math, "Orca-Math word problems (approved pool); digit tokens of the worked solution"),
                ("code", build_code, "CodeAlpaca Python tasks (approved pool); tokens of the reference solution"),
                ("facts", build_facts, "TriviaQA short answers (no context)"),
                ("multilingual", build_multilingual, "English -> de/fr/es/it/pt translation (OPUS-100)"),
                ("general_text", build_general_text, "WikiText paragraph continuation (control only)")]
    skills = {}
    for name, fn, desc in builders:
        rng = random.Random(SEED)
        items = fn(rng)
        if len(items) < N_SEARCH + N_HELDOUT:
            raise RuntimeError(name + ": only " + str(len(items)) + " items")
        skills[name] = {"name": name, "description": desc, "source": "builtin", "items": split(items)}
        print(name, len(items), "items")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(skills, f)
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
