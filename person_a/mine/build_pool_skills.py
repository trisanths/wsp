# Builds the broad skill sets for mine/maps.py. Mining may use any dataset, benchmarks included; each
# skill records its dataset, so compile and heal can avoid training on a benchmark they test on. Fixed seed.
#   .venv/bin/python mine/build_pool_skills.py     -> writes mine/data/pool_skills.json
# Item format is the same as in build_data.py: {"context", "continuation", "digits_only"}.
# A skill whose source does not load is skipped and reported; the others are still written.
import hashlib
import json
import os
import random
import sys

SEED = 0
N_SEARCH = 48
N_HELDOUT = 24
BUFFER = 3000          # rows read from each source before the shuffle
MAX_CONTEXT = 500
MAX_ANSWER = 320
PER_SKILL_SECONDS = 150
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "pool_skills.json")


def stream(dataset, config=None, split="train"):
    import datasets
    if config is None:
        return datasets.load_dataset(dataset, split=split, streaming=True)
    return datasets.load_dataset(dataset, config, split=split, streaming=True)


def take(rows, to_pair, limit=BUFFER):
    # to_pair(row) -> (question, answer) or None
    out = []
    for row in rows:
        pair = to_pair(row)
        if pair is None:
            continue
        q, a = pair[0].strip(), pair[1].strip()
        if len(q) < 15 or len(q) > MAX_CONTEXT or len(a) < 30:
            continue
        out.append((q, a[:MAX_ANSWER]))
        if len(out) >= limit:
            break
    return out


def qa_items(pairs, rng, q_label="Question", a_label="Answer"):
    rng.shuffle(pairs)
    shot = pairs[0]
    shots = q_label + ": " + shot[0] + "\n" + a_label + ": " + shot[1] + "\n\n"
    items = []
    for q, a in pairs[1:1 + N_SEARCH + N_HELDOUT]:
        items.append({"context": shots + q_label + ": " + q + "\n" + a_label + ":", "continuation": " " + a,
                      "digits_only": False})
    return items


def prose_items(texts, rng):
    rng.shuffle(texts)
    items = []
    for t in texts[:N_SEARCH + N_HELDOUT]:
        items.append({"context": t[:200], "continuation": t[200:200 + MAX_ANSWER], "digits_only": False})
    return items


def alpaca(row):
    q = row["instruction"]
    if row.get("input"):
        q = q + " " + row["input"]
    return q, row.get("output") or ""


def wiki_texts(lang):
    texts = []
    for row in stream("wikimedia/wikipedia", "20231101." + lang):
        t = row["text"].strip().replace("\n", " ")
        if len(t) >= 700:
            texts.append(t)
        if len(texts) >= 600:
            break
    return texts


def code_pairs(lang):
    def pick(row):
        if row.get("lang") != lang:
            return None
        return row["problem"][:MAX_CONTEXT - 20], row["solution"]  # task texts are long: keep the start
    return take(stream("ise-uiuc/Magicoder-OSS-Instruct-75K"), pick, limit=300)


def glaive(row):
    chat = row["chat"]
    cut = chat.find("ASSISTANT:")
    if cut < 0:
        return None
    return chat[:cut].replace("USER:", "").strip(), chat[cut + len("ASSISTANT:"):].split("USER:")[0]


# name: (description, dataset text for the record, builder)
QA = {
    "medicine": ("medical flashcard answers", "medalpaca/medical_meadow_medical_flashcards train",
                 lambda: take(stream("medalpaca/medical_meadow_medical_flashcards"),
                              lambda r: (r.get("input") or "", r.get("output") or ""))),
    "clinical_dialogue": ("doctor replies to patient messages", "lavita/ChatDoctor-HealthCareMagic-100k train",
                          lambda: take(stream("lavita/ChatDoctor-HealthCareMagic-100k"),
                                       lambda r: (r.get("input") or "", r.get("output") or ""))),
    "finance": ("finance question answers", "gbharti/finance-alpaca train",
                lambda: take(stream("gbharti/finance-alpaca"), alpaca)),
    "sql": ("SQL query for a question and a table schema", "b-mc2/sql-create-context train",
            lambda: take(stream("b-mc2/sql-create-context"),
                         lambda r: (r["question"] + " Schema: " + r["context"], r["answer"] + " " * 30))),
    "tool_calling": ("assistant turn of a function-calling chat", "glaiveai/glaive-function-calling-v2 train",
                     lambda: take(stream("glaiveai/glaive-function-calling-v2"), glaive)),
    "instruction": ("general instruction following", "yahma/alpaca-cleaned train",
                    lambda: take(stream("yahma/alpaca-cleaned"), alpaca)),
    "creative_writing": ("story continuation from a writing prompt", "euclaise/writingprompts train",
                         lambda: take(stream("euclaise/writingprompts"), lambda r: (r["prompt"], r["story"]))),
}
# MMLU subjects grouped into domains. MMLU is a benchmark; the owner allows any benchmark for mining.
# Compile and heal must not train on a benchmark they test on: each skill records its dataset.
MMLU_DOMAINS = {
    "physics": ["high_school_physics", "college_physics", "conceptual_physics", "astronomy"],
    "chemistry": ["high_school_chemistry", "college_chemistry"],
    "biology": ["high_school_biology", "college_biology", "anatomy", "virology", "medical_genetics"],
    "mathematics_theory": ["abstract_algebra", "college_mathematics", "high_school_mathematics",
                           "high_school_statistics", "formal_logic"],
    "computer_science": ["college_computer_science", "high_school_computer_science", "computer_security",
                         "machine_learning", "electrical_engineering"],
    "law": ["professional_law", "international_law", "jurisprudence"],
    "economics_business": ["high_school_macroeconomics", "high_school_microeconomics", "econometrics",
                           "business_ethics", "management", "marketing", "professional_accounting"],
    "history": ["high_school_us_history", "high_school_world_history", "high_school_european_history",
                "prehistory"],
    "philosophy_ethics": ["philosophy", "moral_disputes", "moral_scenarios", "logical_fallacies", "world_religions"],
    "psychology_social": ["high_school_psychology", "professional_psychology", "sociology", "human_sexuality"],
    "politics_geography": ["high_school_government_and_politics", "us_foreign_policy", "public_relations",
                           "security_studies", "high_school_geography", "global_facts"],
    "clinical_knowledge": ["clinical_knowledge", "college_medicine", "professional_medicine", "nutrition",
                           "human_aging"],
}


def mmlu_pairs():
    # one pass over the test split; answer text is the correct choice
    by_domain = {name: [] for name in MMLU_DOMAINS}
    subject_domain = {}
    for name in MMLU_DOMAINS:
        for subject in MMLU_DOMAINS[name]:
            subject_domain[subject] = name
    for row in stream("cais/mmlu", "all", split="test"):
        name = subject_domain.get(row["subject"])
        if name is None:
            continue
        q = row["question"].strip()
        a = str(row["choices"][int(row["answer"])]).strip()
        if 15 <= len(q) <= MAX_CONTEXT and len(a) >= 2:
            by_domain[name].append((q, a))
    return by_domain


def mmlu_items(pairs, rng):
    rng.shuffle(pairs)
    shots = ""
    for q, a in pairs[:2]:
        shots = shots + "Question: " + q + "\nAnswer: " + a + "\n\n"
    items = []
    for q, a in pairs[2:2 + N_SEARCH + N_HELDOUT]:
        items.append({"context": shots + "Question: " + q + "\nAnswer:", "continuation": " " + a,
                      "digits_only": False})
    return items


CODE_LANGS = ["typescript", "java", "cpp", "csharp", "rust", "php", "swift", "shell"]  # the labels Magicoder has
PROSE_LANGS = {"de": "German", "fr": "French", "es": "Spanish", "zh": "Chinese", "ja": "Japanese", "hi": "Hindi",
               "ar": "Arabic", "ru": "Russian"}


def main():
    plan = []
    for name in QA:
        desc, source, build = QA[name]
        plan.append((name, desc, source, lambda build=build: qa_items(build(), random.Random(SEED))))
    mmlu_cache = {}

    def mmlu_domain(name):
        if not mmlu_cache:
            mmlu_cache.update(mmlu_pairs())
        return mmlu_items(mmlu_cache[name], random.Random(SEED))
    for name in MMLU_DOMAINS:
        plan.append((name, name.replace("_", " ") + " questions, correct answer text",
                     "cais/mmlu all test (subjects: " + ", ".join(MMLU_DOMAINS[name]) + ")",
                     lambda name=name: mmlu_domain(name)))
    for lang in CODE_LANGS:
        plan.append(("code_" + lang, lang + " solutions to programming tasks",
                     "ise-uiuc/Magicoder-OSS-Instruct-75K train (lang=" + lang + ")",
                     lambda lang=lang: qa_items(code_pairs(lang), random.Random(SEED), "# Task", "# Solution")))
    for lang in PROSE_LANGS:
        plan.append(("prose_" + lang, PROSE_LANGS[lang] + " encyclopedia text continuation",
                     "wikimedia/wikipedia 20231101." + lang + " train",
                     lambda lang=lang: prose_items(wiki_texts(lang), random.Random(SEED))))
    plan.append(("science_papers", "arXiv paper text continuation", "ccdv/arxiv-summarization document train",
                 lambda: prose_items([r["article"].strip().replace("\n", " ")
                                      for _, r in zip(range(600), stream("ccdv/arxiv-summarization", "document"))
                                      if len(r["article"]) >= 700], random.Random(SEED))))
    # Resume: keep the skills that an earlier run wrote. Each source gets PER_SKILL_SECONDS, because a
    # stalled network stream would otherwise hang the whole build.
    import signal

    def on_alarm(signum, frame):
        raise TimeoutError("no result in " + str(PER_SKILL_SECONDS) + " s")
    signal.signal(signal.SIGALRM, on_alarm)
    skills = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            skills = json.load(f)
    failed = []
    for name, desc, source, build in plan:
        if name in skills and len(skills[name]["items"]["search"]) == N_SEARCH:
            print("have", name, flush=True)
            continue
        signal.alarm(PER_SKILL_SECONDS)
        try:
            items = build()
        except Exception as e:
            failed.append(name + ": " + repr(e)[:160])
            print("SKIP", failed[-1], flush=True)
            continue
        finally:
            signal.alarm(0)
        if len(items) < N_SEARCH + N_HELDOUT:
            failed.append(name + ": only " + str(len(items)) + " items")
            print("SKIP", failed[-1], flush=True)
            continue
        split = {"search": items[:N_SEARCH], "heldout": items[N_SEARCH:N_SEARCH + N_HELDOUT]}
        digest = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()[:16]
        skills[name] = {"name": name, "description": desc, "source": "pool", "dataset": source, "hash": digest,
                        "items": split}
        print("ok  ", name, digest, flush=True)
        with open(OUT + ".tmp", "w") as f:
            json.dump(skills, f)
        os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, len(skills), "skills,", os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
