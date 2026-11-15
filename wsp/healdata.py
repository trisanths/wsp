"""Text for stitches, routers, and heals that never comes from a benchmark a request is scored on.

Rule (owner, 2026-11-14, clarified the same day): nothing that trains or
selects (a stitch, a router fit, a gate, a heal, a distillation) may use
data from a benchmark the request is SCORED on, in any split. Other
benchmarks and plain corpora are fine and wanted as training data. The
reported set is per request. BANNED is the default reported set (what our
own rounds score). A request that scores other benchmarks passes its own
list as `reported=`.

    from wsp import healdata
    texts = healdata.pool("math", 2000, tokenizer=tokenizer, seed=0)
    texts = healdata.pool("math", 2000, reported=["openai/gsm8k", "ChilleD/SVAMP"])
    items = healdata.practice("knowledge", 50, reported=["mandarjoshi/trivia_qa"])
    print(healdata.describe())          # or: python -m wsp.healdata

A source that is reported, or is derived from a reported benchmark, is
skipped when other sources of the pool remain. A pool with no usable
source raises ValueError.

Practice sets (`practice`) come from other benchmarks, from splits that
no pool trains on. They are for selection and compile-time guards.
"""

import json
import math
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

# Dataset ids (lower case) of benchmarks we report or screen with. This is the DEFAULT reported set.
BANNED = [
    "openai/gsm8k", "gsm8k", "huggingfaceh4/math-500", "eleutherai/hendrycks_math", "hendrycks/competition_math",
    "google-research-datasets/mbpp", "mbpp", "openai/openai_humaneval", "openai_humaneval",
    "openlifescienceai/medmcqa", "medmcqa", "bigbio/med_qa", "gbaker/medqa-usmle-4-options",
    "cais/mmlu", "mmlu", "allenai/ai2_arc", "lukaemon/bbh", "maveriq/bigbenchhard", "juletxara/mgsm",
    "coherelabs/global-mmlu", "cohereforai/global-mmlu", "nguha/legalbench", "casehold/casehold",
    "fastino/fast-decisions", "meta-math/metamathqa", "tiger-lab/mathinstruct",
]

# Config level bans: (dataset id lower case, config) pairs that are a reported benchmark inside a larger repo.
BANNED_CONFIGS = [("coastalcph/lex_glue", "case_hold")]

# Names that mean one benchmark. Used only to match a derived_from entry against a reported list.
FAMILIES = {
    "gsm8k": ["gsm8k", "openai/gsm8k"],
    "math": ["math", "math500", "math-500", "huggingfaceh4/math-500", "eleutherai/hendrycks_math",
             "hendrycks/competition_math", "hendrycks_math", "competition_math"],
    "mbpp": ["mbpp", "google-research-datasets/mbpp"],
    "humaneval": ["humaneval", "openai/openai_humaneval", "openai_humaneval"],
    "medmcqa": ["medmcqa", "openlifescienceai/medmcqa"],
    "mmlu": ["mmlu", "cais/mmlu"],
    "arc": ["arc", "allenai/ai2_arc", "ai2_arc"],
    "svamp": ["svamp", "chilled/svamp"],
    "asdiv": ["asdiv", "eleutherai/asdiv"],
    "aqua": ["aqua", "aqua_rat", "aqua-rat", "deepmind/aqua_rat"],
    "pubmedqa": ["pubmedqa", "qiaojin/pubmedqa"],
    "triviaqa": ["triviaqa", "trivia_qa", "mandarjoshi/trivia_qa"],
    "commonsenseqa": ["commonsenseqa", "commonsense_qa", "tau/commonsense_qa"],
    "openbookqa": ["openbookqa", "allenai/openbookqa"],
    "apps": ["apps", "codeparrot/apps", "mxzoo/apps-introductory-cleaned"],
}

# name: (dataset id, config or None, split, kind). Every source here is usable under the default reported set.
SOURCES = {
    "math": [("microsoft/orca-math-word-problems-200k", None, "train", "qa"),
             ("ChilleD/SVAMP", None, "train", "svamp"),
             ("deepmind/aqua_rat", "raw", "train", "aqua")],
    "code": [("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", "magicoder"),
             ("sahil2801/CodeAlpaca-20k", None, "train", "alpaca"),
             ("iamtarun/python_code_instructions_18k_alpaca", None, "train", "alpaca"),
             ("bigcode/self-oss-instruct-sc2-exec-filter-50k", None, "train", "selfoss"),
             ("mxzoo/apps-introductory-cleaned", None, "train", "apps"),
             ("deepmind/code_contests", None, "train", "contests")],
    "medicine": [("medalpaca/medical_meadow_wikidoc", None, "train", "alpaca"),
                 ("medalpaca/medical_meadow_medical_flashcards", None, "train", "alpaca"),
                 ("qiaojin/PubMedQA", "pqa_artificial", "train", "pubmedqa"),
                 ("lavita/MedQuAD", None, "train", "medquad"),
                 ("FreedomIntelligence/medical-o1-reasoning-SFT", "en", "train", "o1med"),
                 ("ccdv/pubmed-summarization", "document", "train", "abstract")],
    "law": [("coastalcph/lex_glue", "ecthr_a", "train", "legal_text"),
            ("coastalcph/lex_glue", "scotus", "train", "legal_text"),
            ("coastalcph/lex_glue", "ledgar", "train", "classify"),
            ("HFforLegal/case-law", None, "us", "document"),
            ("nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions", None, "train", "alpaca"),
            ("Azzindani/Legal_Corpus_QA_SynDeepThink", None, "train", "qa")],
    "knowledge": [("mandarjoshi/trivia_qa", "rc.nocontext", "train", "trivia"),
                  ("google-research-datasets/nq_open", "nq_open", "train", "nqopen"),
                  ("rajpurkar/squad", "plain_text", "train", "squad"),
                  ("microsoft/wiki_qa", None, "train", "wikiqa")],
    "reasoning": [("tau/commonsense_qa", None, "train", "mc"),
                  ("allenai/openbookqa", "main", "train", "mc"),
                  ("allenai/qasc", None, "train", "mc"),
                  ("ChilleD/StrategyQA", None, "train", "strategyqa"),
                  ("datatune/LogiQA2.0", None, "train", "logiqa")],
    "multilingual": [("Helsinki-NLP/opus-100", "de-en", "train", "translation"),
                     ("Helsinki-NLP/opus-100", "en-es", "train", "translation"),
                     ("Helsinki-NLP/opus-100", "en-fr", "train", "translation"),
                     ("Helsinki-NLP/opus-100", "en-zh", "train", "translation"),
                     ("Helsinki-NLP/opus-100", "en-ru", "train", "translation"),
                     ("CohereLabs/aya_dataset", None, "train", "aya")],
    "general": [("HuggingFaceH4/ultrachat_200k", None, "train_sft", "ultrachat"),
                ("yahma/alpaca-cleaned", None, "train", "alpaca")],
    "prose": [("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text")],
}

# Sources that are usable only when the benchmarks they derive from are not reported. They are
# not in SOURCES because the default reported set excludes them.
EXTRA_SOURCES = {
    "math": [("nvidia/OpenMathInstruct-2", None, "train_1M", "openmath")],
}

# dataset id (lower case) -> benchmarks the set was built from. A source is excluded whenever any of them is reported.
DERIVED_FROM = {
    "nvidia/openmathinstruct-2": ["gsm8k", "math"],
    "meta-math/metamathqa": ["gsm8k", "math"],
    "tiger-lab/mathinstruct": ["gsm8k", "math"],
}

# (dataset id, config, split) -> row count read from the datasets-server /size endpoint on 2026-11-14.
ROWS = {
    ("microsoft/orca-math-word-problems-200k", None, "train"): 200035,
    ("ChilleD/SVAMP", None, "train"): 700,
    ("ChilleD/SVAMP", None, "test"): 300,
    ("EleutherAI/asdiv", "asdiv", "validation"): 2305,
    ("deepmind/aqua_rat", "raw", "train"): 97467,
    ("deepmind/aqua_rat", "raw", "validation"): 254,
    ("nvidia/OpenMathInstruct-2", None, "train_1M"): 1000000,
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train"): 75197,
    ("sahil2801/CodeAlpaca-20k", None, "train"): 20022,
    ("iamtarun/python_code_instructions_18k_alpaca", None, "train"): 18612,
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k", None, "train"): 50661,
    ("mxzoo/apps-introductory-cleaned", None, "train"): 2394,
    ("mxzoo/apps-introductory-cleaned", None, "test"): 726,
    ("deepmind/code_contests", None, "train"): 3762,
    ("medalpaca/medical_meadow_wikidoc", None, "train"): 10000,
    ("medalpaca/medical_meadow_medical_flashcards", None, "train"): 33955,
    ("qiaojin/PubMedQA", "pqa_artificial", "train"): 211269,
    ("qiaojin/PubMedQA", "pqa_labeled", "train"): 1000,
    ("lavita/MedQuAD", None, "train"): 47441,
    ("FreedomIntelligence/medical-o1-reasoning-SFT", "en", "train"): 19704,
    ("ccdv/pubmed-summarization", "document", "train"): 119924,
    ("coastalcph/lex_glue", "ecthr_a", "train"): 9000,
    ("coastalcph/lex_glue", "scotus", "train"): 5000,
    ("coastalcph/lex_glue", "ledgar", "train"): 60000,
    ("HFforLegal/case-law", None, "us"): 282390,
    ("nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions", None, "train"): 4394,
    ("Azzindani/Legal_Corpus_QA_SynDeepThink", None, "train"): 2921,
    ("mandarjoshi/trivia_qa", "rc.nocontext", "train"): 138384,
    ("mandarjoshi/trivia_qa", "rc.nocontext", "validation"): 17944,
    ("google-research-datasets/nq_open", "nq_open", "train"): 87925,
    ("rajpurkar/squad", "plain_text", "train"): 87599,
    ("microsoft/wiki_qa", None, "train"): 20360,
    ("tau/commonsense_qa", None, "train"): 9741,
    ("tau/commonsense_qa", None, "validation"): 1221,
    ("ChilleD/StrategyQA", None, "train"): 1603,
    ("allenai/openbookqa", "main", "train"): 4957,
    ("allenai/openbookqa", "main", "validation"): 500,
    ("allenai/qasc", None, "train"): 8134,
    ("datatune/LogiQA2.0", None, "train"): 63718,
    ("Helsinki-NLP/opus-100", "de-en", "train"): 1000000,
    ("Helsinki-NLP/opus-100", "en-es", "train"): 1000000,
    ("Helsinki-NLP/opus-100", "en-fr", "train"): 1000000,
    ("Helsinki-NLP/opus-100", "en-zh", "train"): 1000000,
    ("Helsinki-NLP/opus-100", "en-ru", "train"): 1000000,
    ("CohereLabs/aya_dataset", None, "train"): 202362,
    ("HuggingFaceH4/ultrachat_200k", None, "train_sft"): 207865,
    ("yahma/alpaca-cleaned", None, "train"): 51760,
    ("Salesforce/wikitext", "wikitext-103-raw-v1", "train"): 1801350,
}

# Sources read as a shuffled stream (large rows or large files). Others are loaded whole, as before.
STREAMED = [
    ("nvidia/OpenMathInstruct-2", None, "train_1M"),
    ("deepmind/code_contests", None, "train"),
    ("HFforLegal/case-law", None, "us"),
    ("ccdv/pubmed-summarization", "document", "train"),
    ("Helsinki-NLP/opus-100", "de-en", "train"),
    ("Helsinki-NLP/opus-100", "en-es", "train"),
    ("Helsinki-NLP/opus-100", "en-fr", "train"),
    ("Helsinki-NLP/opus-100", "en-zh", "train"),
    ("Helsinki-NLP/opus-100", "en-ru", "train"),
]

# Practice sets. Each entry reads a split that no pool trains on, from row 0 of a seeded sample over the whole split.
#   math: SVAMP test (300 rows), ASDiv validation (2305 rows, ASDiv is in no pool), AQuA-RAT validation (254 rows).
#   knowledge: TriviaQA rc.nocontext validation (17944 rows).
#   reasoning: CommonsenseQA validation (1221 rows), OpenBookQA main validation (500 rows).
#   medicine: PubMedQA pqa_labeled (1000 rows; the medicine pool reads pqa_artificial, a different config).
#   code: APPS introductory test split (726 rows, stdin programs only; the code pool reads the train split).
PRACTICE = {
    "math": [("ChilleD/SVAMP", None, "test", "svamp"),
             ("EleutherAI/asdiv", "asdiv", "validation", "asdiv"),
             ("deepmind/aqua_rat", "raw", "validation", "aqua")],
    "knowledge": [("mandarjoshi/trivia_qa", "rc.nocontext", "validation", "trivia")],
    "reasoning": [("tau/commonsense_qa", None, "validation", "mc"),
                  ("allenai/openbookqa", "main", "validation", "mc")],
    "medicine": [("qiaojin/PubMedQA", "pqa_labeled", "train", "pubmedqa")],
    "code": [("mxzoo/apps-introductory-cleaned", None, "test", "apps")],
}

LETTERS = "ABCDEFGHIJ"
LANGUAGE_NAMES = {"de": "German", "es": "Spanish", "fr": "French", "zh": "Chinese", "ru": "Russian",
                  "en": "English", "ar": "Arabic", "hi": "Hindi", "ja": "Japanese"}


# ---------------------------------------------------------------- reported sets


def _lower(name):
    return str(name).lower().strip()


def _reported_list(reported):
    if reported is None:
        return list(BANNED)
    return [_lower(item) for item in reported]


def _direct_match(name, reported_list):
    for banned in reported_list:
        if name == banned or name.endswith("/" + banned):
            return True
    return False


def _family(name):
    for family, members in FAMILIES.items():
        if name == family or name in members or name.endswith("/" + family):
            return family
    return None


def _family_match(name, reported_list):
    """True when name (a derived_from entry) names a benchmark in the reported list."""
    if _direct_match(name, reported_list):
        return True
    family = _family(name)
    if family is None:
        return False
    for item in reported_list:
        if _family(item) == family:
            return True
    return False


def why_reported(dataset_id, reported=None, config=None):
    """Return None when the dataset may train under this reported set, else the reason as text."""
    name = _lower(dataset_id)
    reported_list = _reported_list(reported)
    if _direct_match(name, reported_list):
        return "reported benchmark"
    if reported is None and config is not None and (name, config) in BANNED_CONFIGS:
        return "reported benchmark (config " + str(config) + ")"
    for parent in DERIVED_FROM.get(name, []):
        if _family_match(parent, reported_list):
            return "derived from " + parent + ", which is reported"
    return None


def is_banned(dataset_id, reported=None, config=None):
    """True when a dataset id is in the reported set (any split) or derived from a reported benchmark.

    reported=None means the default set, BANNED.
    """
    return why_reported(dataset_id, reported, config) is not None


def check_sources(dataset_ids, reported=None):
    """Raise ValueError when any id is a reported benchmark. Return the ids."""
    bad = []
    for dataset_id in dataset_ids:
        if is_banned(dataset_id, reported):
            bad.append(dataset_id)
    if len(bad) > 0:
        raise ValueError("training data may not come from a reported benchmark: " + ", ".join(bad))
    return list(dataset_ids)


def _usable(name, reported=None):
    """Return (usable sources, excluded list of (source, reason)) for one pool."""
    if name not in SOURCES:
        raise ValueError("unknown pool: " + str(name) + "; pools: " + ", ".join(sorted(SOURCES)))
    usable = []
    excluded = []
    for source in SOURCES[name] + EXTRA_SOURCES.get(name, []):
        reason = why_reported(source[0], reported, source[1])
        if reason is None:
            usable.append(source)
        else:
            excluded.append((source, reason))
    return usable, excluded


# ---------------------------------------------------------------- renderers


def _options(labels, texts):
    lines = []
    for label, text in zip(labels, texts):
        lines.append(str(label) + ". " + str(text))
    return "\n".join(lines)


def _clean_option(text):
    return re.sub(r"^\(?[A-Ja-j][\).]\s*", "", str(text)).strip()


def _text_of(value):
    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value)
    return str(value)


def row_pair(row, kind, features=None):
    """Render one dataset row as (user, assistant), or None to skip it.

    Real datasets hold odd rows (a LogiQA row without options, a missing
    field, an empty answer). A row the renderer cannot read is skipped,
    never an error, so one bad row cannot stop a compile.
    """
    try:
        return _row_pair_strict(row, kind, features)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None


def _row_pair_strict(row, kind, features=None):
    """Return (user, assistant) for one dataset row, or None to skip it. assistant is None for plain text.

    features is the dataset's features mapping, needed only by kind "classify" (label names).
    """
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
    if kind == "svamp":
        user = (row["Body"].strip() + " " + row["Question"].strip()).strip()
        answer = str(row["Answer"]).strip()
        if len(answer) == 0:
            return None
        return user, "Equation: " + str(row["Equation"]).strip() + "\nThe answer is " + answer + "."
    if kind == "asdiv":
        user = (row["body"].strip() + " " + row["question"].strip()).strip()
        return user, "Equation: " + str(row["formula"]).strip() + "\nThe answer is " + str(row["answer"]).strip() + "."
    if kind == "aqua":
        texts = [_clean_option(item) for item in row["options"]]
        labels = list(LETTERS[:len(texts)])
        if row["correct"] not in labels:
            return None
        gold = texts[labels.index(row["correct"])]
        user = row["question"].strip() + "\n" + _options(labels, texts)
        return user, row["rationale"].strip() + "\nThe answer is " + row["correct"] + ". " + gold
    if kind == "openmath":
        if not row.get("problem") or not row.get("generated_solution"):
            return None
        return row["problem"], row["generated_solution"]
    if kind == "selfoss":
        if not row.get("instruction") or not row.get("response"):
            return None
        return row["instruction"], row["response"]
    if kind == "apps":
        solutions = row.get("solutions")
        if isinstance(solutions, str):
            try:
                solutions = json.loads(solutions)
            except ValueError:
                return None
        if not solutions:
            return None
        user = row["question"].strip()
        if row.get("starter_code"):
            user = user + "\n\nStarter code:\n" + row["starter_code"]
        return user, "```python\n" + solutions[0].strip() + "\n```"
    if kind == "contests":
        solutions = row.get("solutions") or {}
        languages = solutions.get("language") or []
        sources = solutions.get("solution") or []
        for language, source in zip(languages, sources):
            if language == 3:
                return row["description"].strip(), "```python\n" + source.strip() + "\n```"
        return None
    if kind == "pubmedqa":
        contexts = row["context"]["contexts"] if isinstance(row["context"], dict) else row["context"]
        user = "Question: " + row["question"].strip() + "\n\nContext:\n" + _text_of(contexts)
        decision = row.get("final_decision")
        if not row.get("long_answer") or decision is None:
            return None
        return user, row["long_answer"].strip() + "\nAnswer: " + str(decision)
    if kind == "medquad":
        if not row.get("question") or not row.get("answer"):
            return None
        return row["question"], row["answer"]
    if kind == "o1med":
        if not row.get("Question") or not row.get("Response"):
            return None
        return row["Question"], (row.get("Complex_CoT") or "").strip() + "\n\n" + row["Response"]
    if kind == "abstract":
        text = (row.get("abstract") or "").strip()
        if len(text) < 200:
            return None
        return text, None
    if kind == "legal_text":
        text = _text_of(row["text"]).strip()
        if len(text) < 200:
            return None
        return text, None
    if kind == "document":
        text = (row.get("document") or "").strip()
        if len(text) < 200:
            return None
        return text[:6000], None
    if kind == "classify":
        names = None
        if features is not None and "label" in features and hasattr(features["label"], "names"):
            names = features["label"].names
        label = row["label"]
        name = names[label] if names is not None else "label " + str(label)
        return "Name the topic of this contract provision.\n\n" + row["text"].strip(), name
    if kind == "trivia":
        answer = row["answer"]
        value = answer["value"] if isinstance(answer, dict) else answer
        if not value:
            return None
        return row["question"], str(value)
    if kind == "nqopen":
        answers = row["answer"]
        if not answers:
            return None
        return row["question"], str(answers[0])
    if kind == "squad":
        answers = row["answers"]["text"]
        if len(answers) == 0:
            return None
        return "Context: " + row["context"].strip() + "\n\nQuestion: " + row["question"].strip(), answers[0]
    if kind == "wikiqa":
        if row.get("label") != 1:
            return None
        return row["question"], row["answer"]
    if kind == "mc":
        question = row.get("question") or row.get("question_stem")
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        key = row["answerKey"]
        if key not in labels:
            return None
        gold = texts[labels.index(key)]
        return question.strip() + "\n" + _options(labels, texts), key + ". " + gold
    if kind == "strategyqa":
        word = "Yes" if row["answer"] else "No"
        facts = str(row.get("facts") or "").strip()
        return row["question"], (facts + "\n" if facts else "") + "The answer is " + word + "."
    if kind == "logiqa":
        data = json.loads(row["text"]) if isinstance(row.get("text"), str) else row
        texts = list(data["options"])
        labels = list(LETTERS[:len(texts)])
        index = int(data["answer"])
        user = data["text"].strip() + "\n\n" + data["question"].strip() + "\n" + _options(labels, texts)
        return user, labels[index] + ". " + texts[index]
    if kind == "translation":
        pair = row["translation"]
        codes = sorted(pair.keys())
        other = codes[0] if codes[1] == "en" else codes[1]
        if "en" not in pair:
            return None
        if len(pair["en"]) % 2 == 0:
            source, target, source_code, target_code = pair[other], pair["en"], other, "en"
        else:
            source, target, source_code, target_code = pair["en"], pair[other], "en", other
        head = "Translate from " + LANGUAGE_NAMES.get(source_code, source_code) + " to "
        return head + LANGUAGE_NAMES.get(target_code, target_code) + ":\n\n" + source, target
    if kind == "aya":
        if not row.get("inputs") or not row.get("targets"):
            return None
        return row["inputs"], row["targets"]
    raise ValueError("unknown row kind: " + str(kind))


def render(tokenizer, user, assistant):
    """Chat-format one pair with the tokenizer's template. Plain text passes through."""
    if assistant is None:
        return user
    if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
        return user + "\n\n" + assistant
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ---------------------------------------------------------------- loading


def _load(dataset_id, config, split, stream=False, seed=0):
    """Return rows of one source: a list-like dataset, or a shuffled stream. Tests replace this."""
    from datasets import load_dataset

    if stream:
        if config is None:
            rows = load_dataset(dataset_id, split=split, streaming=True)
        else:
            rows = load_dataset(dataset_id, config, split=split, streaming=True)
        return rows.shuffle(seed=seed, buffer_size=2000)
    if config is None:
        return load_dataset(dataset_id, split=split)
    return load_dataset(dataset_id, config, split=split)


def _features(rows):
    return getattr(rows, "features", None)


def pairs(name, count, seed=0, skip=0, reported=None):
    """Return count (user, assistant) pairs of one pool, after skipping skip pairs. Seeded order.

    Sources that are reported (or derived from a reported benchmark) are
    skipped. ValueError when no source of the pool is left. The list is
    shorter than count only when every source is spent; callers that need
    an exact count must check the length.
    """
    sources, excluded = _usable(name, reported)
    if len(sources) == 0:
        reasons = []
        for source, reason in excluded:
            reasons.append(source[0] + " (" + reason + ")")
        raise ValueError("pool " + name + " has no usable source for this reported set: " + ", ".join(reasons))
    ids = []
    for source in sources:
        ids.append(source[0])
    check_sources(ids, reported)
    want = count + skip
    streams = []
    for dataset_id, config, split, kind in sources:
        streams.append(_source_pairs(dataset_id, config, split, kind, seed, want))
    # Round robin over the sources, each in its own fixed seeded order. A small source runs out
    # and the others keep going, so a pool is short only when every source is spent. The order
    # does not depend on count, so pairs(name, n) is a prefix of pairs(name, m) for m > n, and
    # two calls with different skip values give disjoint slices.
    out = []
    live = list(streams)
    while len(out) < want and len(live) > 0:
        still = []
        for stream in live:
            if len(out) >= want:
                still.append(stream)
                continue
            pair = next(stream, None)
            if pair is None:
                continue
            out.append(pair)
            still.append(stream)
        live = still
    return out[skip:skip + count]


def _source_pairs(dataset_id, config, split, kind, seed, want):
    """Yield the readable (user, assistant) pairs of one source in a fixed seeded order."""
    streamed = (dataset_id, config, split) in STREAMED
    rows = _load(dataset_id, config, split, stream=streamed, seed=seed)
    features = _features(rows)
    if streamed:
        # A stream cannot be indexed. Read on until enough pairs came out, with a bound on rows read.
        for seen, row in enumerate(rows):
            if seen > (want + 1) * 40:
                return
            pair = row_pair(row, kind, features)
            if pair is not None:
                yield pair
        return
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    for index in order:
        pair = row_pair(rows[index], kind, features)
        if pair is not None:
            yield pair


def pool(name, count, tokenizer=None, seed=0, skip=0, reported=None):
    """Return count training texts of one pool, chat-formatted when a tokenizer with a template is given.

    skip lets two uses (for example a stitch and a heal) take disjoint
    parts of the same pool with the same seed.
    """
    texts = []
    for user, assistant in pairs(name, count, seed=seed, skip=skip, reported=reported):
        texts.append(render(tokenizer, user, assistant))
    return texts


def mix(names, per_pool, tokenizer=None, seed=0, skip=0, reported=None):
    """Return a shuffled mix of per_pool texts from each named pool."""
    texts = []
    for name in names:
        for text in pool(name, per_pool, tokenizer=tokenizer, seed=seed, skip=skip, reported=reported):
            texts.append(text)
    random.Random(seed + 2).shuffle(texts)
    return texts


# ---------------------------------------------------------------- practice sets


def _number_text(value):
    """Return the leading number of an answer string such as '9 (apples)' as text, or None."""
    match = re.search(r"-?\d[\d,]*\.?\d*|-?\.\d+", str(value))
    if match is None:
        return None
    return match.group(0).replace(",", "").rstrip(".")


def practice_item(row, kind, source_id):
    """Return one practice item for a row, or None to skip it."""
    base = {"choices": None, "source": source_id}
    if kind == "svamp":
        answer = _number_text(row["Answer"])
        if answer is None:
            return None
        prompt = (row["Body"].strip() + " " + row["Question"].strip()).strip()
        return dict(base, prompt=prompt, answer=answer, kind="number")
    if kind == "asdiv":
        answer = _number_text(row["answer"])
        if answer is None:
            return None
        prompt = (row["body"].strip() + " " + row["question"].strip()).strip()
        return dict(base, prompt=prompt, answer=answer, kind="number")
    if kind == "aqua":
        texts = [_clean_option(item) for item in row["options"]]
        labels = list(LETTERS[:len(texts)])
        if row["correct"] not in labels:
            return None
        choices = []
        for label, text in zip(labels, texts):
            choices.append(label + ". " + text)
        return dict(base, prompt=row["question"].strip(), answer=row["correct"], kind="choice", choices=choices)
    if kind == "mc":
        question = row.get("question") or row.get("question_stem")
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        if row["answerKey"] not in labels:
            return None
        choices = []
        for label, text in zip(labels, texts):
            choices.append(label + ". " + text)
        return dict(base, prompt=question.strip(), answer=row["answerKey"], kind="choice", choices=choices)
    if kind == "trivia":
        answer = row["answer"]
        value = answer.get("value") if isinstance(answer, dict) else None
        if not value:
            return None
        aliases = list(answer.get("aliases") or [])
        if value not in aliases:
            aliases.append(value)
        return dict(base, prompt=row["question"].strip(), answer=str(value), kind="text", aliases=aliases)
    if kind == "pubmedqa":
        decision = row.get("final_decision")
        if decision not in ("yes", "no", "maybe"):
            return None
        contexts = row["context"]["contexts"] if isinstance(row["context"], dict) else row["context"]
        prompt = "Context:\n" + _text_of(contexts) + "\n\nQuestion: " + row["question"].strip()
        return dict(base, prompt=prompt, answer=decision, kind="choice", choices=["yes", "no", "maybe"])
    if kind == "apps":
        if row.get("starter_code"):
            return None
        try:
            tests = json.loads(row["input_output"]) if isinstance(row["input_output"], str) else row["input_output"]
            inputs = list(tests["inputs"])
            outputs = list(tests["outputs"])
        except (ValueError, KeyError, TypeError):
            return None
        if len(inputs) == 0 or len(inputs) != len(outputs):
            return None
        if not all(isinstance(item, str) for item in inputs + outputs):
            return None
        if sum(len(item) for item in inputs[:5] + outputs[:5]) > 20000:
            return None
        return dict(base, prompt=row["question"].strip(), answer="", kind="code",
                    tests={"inputs": inputs[:5], "outputs": outputs[:5]})
    raise ValueError("unknown practice kind: " + str(kind))


def practice(skill, count, reported=None, seed=0):
    """Return up to count practice items for a skill from non-reported benchmarks.

    Each item is {"prompt", "answer", "kind", "choices", "source"}; kind is
    "number", "choice", "text", or "code". The rows come from the splits
    listed in PRACTICE, which no pool trains on. ValueError when the skill
    has no practice set or every source of it is reported.
    """
    if skill not in PRACTICE:
        raise ValueError("no practice set for skill " + str(skill) + "; skills: " + ", ".join(sorted(PRACTICE)))
    active = []
    for source in PRACTICE[skill]:
        if why_reported(source[0], reported, source[1]) is None:
            active.append(source)
    if len(active) == 0:
        raise ValueError("every practice source of " + skill + " is in the reported set")
    each = int(math.ceil(count / float(len(active))))
    items = []
    for dataset_id, config, split, kind in active:
        rows = _load(dataset_id, config, split, stream=False, seed=seed)
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        taken = 0
        for index in order:
            if taken >= each:
                break
            item = practice_item(rows[index], kind, dataset_id)
            if item is None:
                continue
            items.append(item)
            taken += 1
    random.Random(seed + 1).shuffle(items)
    return items[:count]


def _normal(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _choice_prediction(text, choices):
    """Return the chosen letter (or yes/no/maybe word) in a model output, or None."""
    words = ("yes", "no", "maybe")
    if choices is not None and list(choices) == list(words):
        found = re.findall(r"\b(yes|no|maybe)\b", str(text).lower())
        return found[-1] if found else None
    patterns = [r"(?i:answer is)[:\s]*\(?\**([A-J])\b", r"(?i:answer)[:\s]+\(?\**([A-J])\b", r"\\boxed\{\(?([A-J])\)?\}",
                r"^\s*\(?([A-J])[\).:]?\s*$", r"^\s*\(?([A-J])[\).:]\s"]
    for pattern in patterns:
        found = re.findall(pattern, str(text), flags=re.MULTILINE)
        if found:
            return found[-1]
    return None


def _number_prediction(text):
    try:
        from wsp import skills_eval
        return skills_eval.gsm8k_prediction(str(text)), skills_eval.same_number
    except Exception:
        found = re.findall(r"-?[\d,]*\.?\d+", str(text))
        value = None
        if found:
            try:
                value = float(found[-1].replace(",", "").rstrip("."))
            except ValueError:
                value = None

        def same(a, b):
            return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))
        return value, same


def _code_of(text):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", str(text), flags=re.DOTALL)
    if blocks:
        return blocks[-1]
    return str(text)


def check_answer(item, text):
    """True when the model output text answers the practice item correctly.

    number: the predicted number (skills_eval.gsm8k_prediction) equals the gold number.
    choice: the chosen letter, or the yes, no, or maybe word, equals the gold answer.
    text: a gold alias appears in the output (normalized words).
    code: the program in the output passes the stored stdin and stdout tests in the skills_eval sandbox.
    """
    kind = item["kind"]
    if kind == "number":
        predicted, same = _number_prediction(text)
        try:
            gold = float(str(item["answer"]).replace(",", ""))
        except ValueError:
            return False
        return same(predicted, gold)
    if kind == "choice":
        predicted = _choice_prediction(text, item.get("choices"))
        return predicted is not None and predicted.lower() == str(item["answer"]).lower()
    if kind == "text":
        output = " " + _normal(text) + " "
        for alias in item.get("aliases") or [item["answer"]]:
            wanted = _normal(alias)
            if wanted and " " + wanted + " " in output:
                return True
        return False
    if kind == "code":
        from wsp import skills_eval

        tests = [(a, b) for a, b in zip(item["tests"]["inputs"], item["tests"]["outputs"])]
        program = ("import io, sys\n_code = " + repr(_code_of(text)) + "\n_tests = " + repr(tests) + "\n"
                   "for _given, _want in _tests:\n"
                   "    sys.stdin = io.StringIO(_given)\n"
                   "    _buffer = io.StringIO()\n"
                   "    _real = sys.stdout\n"
                   "    sys.stdout = _buffer\n"
                   "    try:\n"
                   "        exec(compile(_code, 'solution', 'exec'), {'__name__': '__main__'})\n"
                   "    finally:\n"
                   "        sys.stdout = _real\n"
                   "    assert _buffer.getvalue().split() == _want.split()\n")
        return bool(skills_eval.run_program(program, timeout=10.0)["passed"])
    raise ValueError("unknown item kind: " + str(kind))


# ---------------------------------------------------------------- description and network sampling


def describe(reported=None):
    """Return {pool: {"usable": [...], "excluded": [...], "practice": [...]}} for a reported set.

    usable and practice entries are dicts with id, config, split, kind, rows (None when unknown).
    excluded entries add the reason.
    """
    out = {}
    for name in SOURCES:
        sources, excluded = _usable(name, reported)
        entry = {"usable": [], "excluded": [], "practice": []}
        for dataset_id, config, split, kind in sources:
            entry["usable"].append({"id": dataset_id, "config": config, "split": split, "kind": kind,
                                    "rows": ROWS.get((dataset_id, config, split))})
        for (dataset_id, config, split, kind), reason in excluded:
            entry["excluded"].append({"id": dataset_id, "config": config, "split": split, "kind": kind,
                                      "rows": ROWS.get((dataset_id, config, split)), "reason": reason})
        for dataset_id, config, split, kind in PRACTICE.get(name, []):
            reason = why_reported(dataset_id, reported, config)
            item = {"id": dataset_id, "config": config, "split": split, "kind": kind,
                    "rows": ROWS.get((dataset_id, config, split))}
            if reason is None:
                entry["practice"].append(item)
            else:
                item["reason"] = reason
                entry["excluded"].append(item)
        out[name] = entry
    return out


def sample_rows(dataset_id, config, split, count=3):
    """Return count rows of a source through the datasets-server /rows endpoint (no datasets library needed)."""
    query = {"dataset": dataset_id, "config": config or "default", "split": split, "offset": 0, "length": count}
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(query)
    data = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = json.load(response)
            break
        except urllib.error.URLError:
            if attempt == 4:
                raise
            time.sleep(8 * (attempt + 1))
    rows = []
    for item in data["rows"]:
        rows.append(item["row"])
    return rows


def _format(entry):
    config = "" if entry["config"] is None else " [" + entry["config"] + "]"
    rows = "" if entry["rows"] is None else ", " + str(entry["rows"]) + " rows"
    text = "  " + entry["id"] + config + " " + entry["split"] + " (" + entry["kind"] + rows + ")"
    if "reason" in entry:
        text = text + ": " + entry["reason"]
    return text


def main(argv=None):
    import sys

    arguments = sys.argv[1:] if argv is None else argv
    reported = arguments if len(arguments) > 0 else None
    shown = "default (BANNED)" if reported is None else ", ".join(reported)
    print("reported set: " + shown)
    for name, entry in describe(reported).items():
        print(name)
        for line in entry["usable"]:
            print(_format(line))
        for line in entry["practice"]:
            print(_format(line).replace("  ", "  practice ", 1))
        for line in entry["excluded"]:
            print(_format(line).replace("  ", "  excluded ", 1))


if __name__ == "__main__":
    main()
