"""Data and scoring helpers for the language write-in demo (docs/write_in_language.md).

Everything here is plain Python plus torch for scoring, so the Mac can test it on tiny random models.

Eval sets (never trained or selected on):
  Belebele test, 900 items per language, 10 languages (facebook/belebele, data/<lang>.jsonl).
  MMMLU held-out: 1000 question indexes, same in all five languages (openai/MMMLU, rows follow the MMLU test order).
  English MMLU held-out 2000 (the xfam split: Random(0) order, items 1000 to 2999).
Dev slice for learning-rate picks: 200 other MMMLU question indexes per language. Not an eval set.
Training: MMMLU questions outside held-out, dev and the xfam English dev and held-out indexes, XNLI train recast as
three-option items, and the English xfam pool.
"""

import csv
import json
import random

import torch

LETTERS = "ABCD"
N_MMLU_TEST = 14042

BELEBELE_TARGET = ["arb_Arab", "jpn_Jpan", "zho_Hans", "kor_Hang", "hin_Deva"]
BELEBELE_ENGLISH = ["eng_Latn"]
BELEBELE_TRANSFER = ["spa_Latn", "fra_Latn", "ben_Beng", "swh_Latn"]
BELEBELE_ALL = BELEBELE_ENGLISH + BELEBELE_TARGET + BELEBELE_TRANSFER

MMMLU_FILES = {"ar": "AR-XY", "ja": "JA-JP", "zh": "ZH-CN", "ko": "KO-KR", "hi": "HI-IN"}
XNLI_LANGS = ["ar", "zh", "hi"]
WIKI_LANGS = {"en": "en", "ar": "ar", "ja": "ja", "zh": "zh", "ko": "ko", "hi": "hi"}
WIKI_CHARS = {"en": 1200, "fr": 1000, "ar": 900, "hi": 900, "ko": 800, "ja": 500, "zh": 500}

HELD_N = 1000
DEV_N = 200
TRAIN_MAX_TOKENS = 512
EVAL_MAX_TOKENS = 4096


def mc_prompt(question, choices):
    """The MMLU prompt of modal_round6.mmlu_prompt, so every source shares one format."""
    lines = ["Question: " + str(question).strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def belebele_prompt(row):
    options = [row["mc_answer%d" % i] for i in range(1, 5)]
    lines = ["Passage: " + row["flores_passage"].strip(), "Question: " + row["question"].strip()]
    for index, option in enumerate(options):
        lines.append(LETTERS[index] + ". " + str(option).strip())
    return "\n".join(lines) + "\nAnswer:"


def belebele_items(path):
    """[prompt, gold] for one Belebele jsonl file (test only: the file holds 900 rows)."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items.append([belebele_prompt(row), int(row["correct_answer_num"]) - 1])
    return items


def mmlu_order(n=N_MMLU_TEST):
    """The xfam shuffle of MMLU test rows: first 1000 are English dev, 1000 to 2999 English held-out."""
    order = list(range(n))
    random.Random(0).shuffle(order)
    return order


def split_indexes(n=N_MMLU_TEST, seed=2026):
    """Question indexes into the MMLU test order.

    Returns a dict: en_dev, en_held (the xfam English sets), held and dev (MMMLU), train (everything else).
    Held-out and dev are drawn from rows outside the xfam English dev and held-out, so a held-out question has
    neither its translations nor its English original in any training set or in the English guard.
    """
    order = mmlu_order(n)
    en_dev = order[:1000]
    en_held = order[1000:3000]
    used = set(en_dev) | set(en_held)
    rest = [i for i in range(n) if i not in used]
    rng = random.Random(seed)
    rng.shuffle(rest)
    held = sorted(rest[:HELD_N])
    dev = sorted(rest[HELD_N:HELD_N + DEV_N])
    train = sorted(rest[HELD_N + DEV_N:])
    return {"en_dev": en_dev, "en_held": en_held, "held": held, "dev": dev, "train": train}


def read_mmmlu(path):
    """Rows of one MMMLU csv as dicts with question, choices, answer (0 to 3) and subject. Row i is MMLU test row i."""
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if len(r["Answer"].strip()) != 1 or r["Answer"].strip() not in LETTERS:
                rows.append(None)
                continue
            rows.append({"question": r["Question"], "choices": [r["A"], r["B"], r["C"], r["D"]],
                         "answer": LETTERS.index(r["Answer"].strip()), "subject": r["Subject"]})
    return rows


def check_alignment(mmmlu_rows, mmlu_test):
    """Fraction of rows whose subject and answer letter equal the English MMLU test row at the same position."""
    n = min(len(mmmlu_rows), len(mmlu_test))
    both = same = 0
    for i in range(n):
        r, e = mmmlu_rows[i], mmlu_test[i]
        if r is None:
            continue
        both += 1
        same += int(r["subject"] == e["subject"] and r["answer"] == int(e["answer"]))
    return same / max(1, both)


def mmmlu_items(rows, indexes, english=None, require_same_answer=False):
    """[prompt, gold] for the given row indexes; with english, drop rows whose answer differs from the original."""
    out = []
    for i in indexes:
        r = rows[i]
        if r is None:
            continue
        if require_same_answer and english is not None and r["answer"] != int(english[i]["answer"]):
            continue
        out.append([mc_prompt(r["question"], r["choices"]), r["answer"]])
    return out


XNLI_OPTIONS = ["The hypothesis follows from the premise.", "The hypothesis may or may not be true.",
                "The hypothesis contradicts the premise."]


def xnli_items(rows, n, seed, max_chars=900):
    """Recast XNLI train rows (dicts with premise, hypothesis, label 0 entailment, 1 neutral, 2 contradiction) as
    three-option items. Option order is shuffled per item so no letter is favoured. Returns [prompt, gold]."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    out = []
    for i in order:
        r = rows[i]
        if r["label"] not in (0, 1, 2) or not r["premise"] or not r["hypothesis"]:
            continue
        if len(r["premise"]) + len(r["hypothesis"]) > max_chars:
            continue
        perm = [0, 1, 2]
        rng.shuffle(perm)
        options = [XNLI_OPTIONS[p] for p in perm]
        question = "Premise: %s\nHypothesis: %s\nHow does the hypothesis relate to the premise?" % (
            r["premise"].strip(), r["hypothesis"].strip())
        out.append([mc_prompt(question, options), perm.index(r["label"])])
        if len(out) >= n:
            break
    return out


def restrict_soft(soft, n_options):
    """Zero the letters beyond n_options in a donor distribution and renormalise."""
    kept = [p if j < n_options else 0.0 for j, p in enumerate(soft)]
    total = sum(kept)
    if total <= 0:
        return [1.0 / n_options if j < n_options else 0.0 for j in range(len(soft))]
    return [round(p / total, 5) for p in kept]


def wiki_chunks(texts, lang, n, seed, min_chars=200):
    """Pick n plain-text chunks from article strings: the start of each article cut to WIKI_CHARS[lang]."""
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    size = WIKI_CHARS[lang]
    out = []
    for i in order:
        t = " ".join(texts[i].split())
        if len(t) < min_chars:
            continue
        out.append(t[:size])
        if len(out) >= n:
            break
    return out


def filter_items(items, banned, tok, max_tokens):
    """Drop items whose prompt is in banned, repeats, or is longer than max_tokens host tokens. Returns (kept, counts)."""
    seen = set()
    kept = []
    counts = {"in": len(items), "banned": 0, "duplicate": 0, "too_long": 0}
    lengths = token_lengths(tok, [p for p, _ in items])
    for (prompt, gold), n in zip(items, lengths):
        if prompt in banned:
            counts["banned"] += 1
        elif prompt in seen:
            counts["duplicate"] += 1
        elif n > max_tokens:
            counts["too_long"] += 1
        else:
            seen.add(prompt)
            kept.append([prompt, gold])
    counts["kept"] = len(kept)
    return kept, counts


def token_lengths(tok, prompts, chunk=2000):
    out = []
    for s in range(0, len(prompts), chunk):
        out.extend(len(x) for x in tok(prompts[s:s + chunk], add_special_tokens=True)["input_ids"])
    return out


# ------------------------------------------------------------------ scoring


def _letter_ids(tok):
    return [tok(" " + c, add_special_tokens=False).input_ids[-1] for c in LETTERS]


def score_long(model, tok, items, max_tokens=EVAL_MAX_TOKENS, budget=16384, return_preds=False):
    """Accuracy by the argmax over ' A' to ' D' at the last prompt token, for prompts of any length.

    Items are sorted by length and batched under a padded-token budget. A prompt longer than max_tokens is cut
    from the left (the question and options stay). Returns a dict: acc, n, truncated, max_len (before the cut),
    mean_len, and preds when asked. Only the last position's logits are computed.
    """
    items = [tuple(x) for x in items]
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _letter_ids(tok)
    device = next(model.parameters()).device
    lengths = token_lengths(tok, [p for p, _ in items])
    order = sorted(range(len(items)), key=lambda i: lengths[i])
    preds = [None] * len(items)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end + 1 - start) * min(lengths[order[end]], max_tokens) <= budget:
            end += 1
        idx = order[start:end]
        enc = tok([items[i][0] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_tokens).to(device)
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()[:, ids]
        for i, row in zip(idx, logits):
            preds[i] = int(row.argmax())
        start = end
    correct = sum(int(p == g) for p, (_, g) in zip(preds, items))
    out = {"acc": correct / max(1, len(items)), "n": len(items),
           "truncated": sum(int(n > max_tokens) for n in lengths), "max_len": max(lengths) if lengths else 0,
           "mean_len": sum(lengths) / max(1, len(lengths))}
    if return_preds:
        out["preds"] = preds
    return out


def score_all(model, tok, evals, **kw):
    """{name: score_long dict} for every eval set."""
    return {name: score_long(model, tok, items, **kw) for name, items in evals.items()}


def pick_letter(model, tok, prompt):
    tok.padding_side = "left"
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc, logits_to_keep=1).logits[0, -1].float()[_letter_ids(tok)]
    return LETTERS[int(logits.argmax())]


def demo_outputs(model, tok, prompts, new_tokens=60):
    """Raw greedy outputs on fixed prompts: the letter picked at the answer position and the first tokens generated."""
    device = next(model.parameters()).device
    rows = []
    for name, prompt in prompts.items():
        letter = pick_letter(model, tok, prompt)
        tok.padding_side = "left"
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 max_new_tokens=new_tokens, do_sample=False)
        text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        rows.append({"prompt_id": name, "letter": letter, "generated": text})
    return rows


# Six fixed prompts for the demo, written for this track and not taken from Belebele. Gold answer is C in each.


def _demo(passage, question, options):
    lines = ["Passage: " + passage, "Question: " + question]
    for i, o in enumerate(options):
        lines.append(LETTERS[i] + ". " + o)
    return "\n".join(lines) + "\nAnswer:"


DEMO_PROMPTS = {
    "en": _demo("The village library opens at nine in the morning and closes at six in the evening. On Fridays it "
                "stays open until eight so that people who work late can borrow books.",
                "When does the library close on Fridays?",
                ["At six in the evening", "At nine in the morning", "At eight in the evening", "At noon"]),
    "ar": _demo("تفتح مكتبة القرية أبوابها في التاسعة صباحا وتغلق في السادسة مساء. وفي أيام الجمعة تبقى مفتوحة حتى "
                "الثامنة مساء حتى يتمكن من يعملون في وقت متأخر من استعارة الكتب.",
                "متى تغلق المكتبة في أيام الجمعة؟",
                ["في السادسة مساء", "في التاسعة صباحا", "في الثامنة مساء", "عند الظهر"]),
    "ja": _demo("村の図書館は朝九時に開き、夕方六時に閉まります。金曜日は、遅くまで働く人が本を借りられるように、夜八時まで開いています。",
                "金曜日、図書館は何時に閉まりますか。",
                ["夕方六時", "朝九時", "夜八時", "正午"]),
    "zh": _demo("村里的图书馆早上九点开门，晚上六点关门。星期五开到晚上八点，方便工作到很晚的人借书。",
                "星期五图书馆几点关门？",
                ["晚上六点", "早上九点", "晚上八点", "中午十二点"]),
    "ko": _demo("마을 도서관은 아침 아홉 시에 열고 저녁 여섯 시에 닫습니다. 금요일에는 늦게까지 일하는 사람들이 책을 빌릴 수 있도록 "
                "저녁 여덟 시까지 문을 엽니다.",
                "금요일에 도서관은 몇 시에 닫습니까?",
                ["저녁 여섯 시", "아침 아홉 시", "저녁 여덟 시", "정오"]),
    "hi": _demo("गाँव का पुस्तकालय सुबह नौ बजे खुलता है और शाम छह बजे बंद हो जाता है। शुक्रवार को यह रात आठ बजे तक खुला रहता है "
                "ताकि देर तक काम करने वाले लोग किताबें ले सकें।",
                "शुक्रवार को पुस्तकालय कब बंद होता है?",
                ["शाम छह बजे", "सुबह नौ बजे", "रात आठ बजे", "दोपहर बारह बजे"]),
}
DEMO_GOLD = "C"


# ------------------------------------------------------------------ medical track (data version v2med)
# Docs: docs/demo_med_multilingual.md. Eval sets nothing trains or selects on, in any language:
#   mmmlu_med_<lang>  every MMMLU row of the six MED6 subjects in Hindi, French, Japanese
#   mmlu_med_en       the same questions in English MMLU test
#   bele_*            Belebele test, 900 per language
#   medmcqa_val       MedMCQA validation 1000 (shuffle seed 7), medqa_test  MedQA USMLE test, mmlu_held  English guard 2000
# Dev slice for learning-rate picks and for choosing the graft seed to save: dev_mmmlu_<lang> (v1 dev indexes minus
# MED6 subjects) and dev_medmcqa (MedMCQA train items held out of the pool).

MED6_SUBJECTS = ["clinical_knowledge", "college_medicine", "professional_medicine", "anatomy", "medical_genetics",
                 "college_biology"]
MED_LANGS = {"hi": "HI-IN", "fr": "FR-FR", "ja": "JA-JP"}
MED_BELEBELE = ["hin_Deva", "fra_Latn", "jpn_Jpan", "eng_Latn"]
MED_XNLI = ["hi", "fr"]
MED_WIKI = {"en": "en", "hi": "hi", "fr": "fr", "ja": "ja"}
DEV_MED_N = 300


def med_split(subjects, v1=None):
    """Index sets of the MMLU test order for the medical track.

    subjects is the subject of each English MMLU test row. Returns dict: med (rows of the six MED6 subjects, every
    language's eval), dev (v1 dev minus MED6), train (v1 train minus MED6). The v1 held-out indexes, the xfam English
    dev and held-out indexes and every MED6 row are in no training set.
    """
    v1 = v1 or split_indexes(len(subjects))
    wanted = set(MED6_SUBJECTS)
    med = [i for i, s in enumerate(subjects) if s in wanted]
    medset = set(med)
    return {"med": med, "dev": [i for i in v1["dev"] if i not in medset],
            "train": [i for i in v1["train"] if i not in medset], "en_held": v1["en_held"], "en_dev": v1["en_dev"],
            "v1_held": v1["held"]}


def usable_train_rows(rows, english_subjects, indexes):
    """Indexes whose translated row exists and whose own subject, like the English subject, is not a MED6 subject."""
    wanted = set(MED6_SUBJECTS)
    return [i for i in indexes if rows[i] is not None and rows[i]["subject"] not in wanted
            and english_subjects[i] not in wanted]


def jsonl_medqa_items(lines):
    """[prompt, gold] for GBaker/MedQA-USMLE-4-options jsonl text lines (question, options dict, answer_idx)."""
    out = []
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        opts = r["options"]
        choices = [opts[k] for k in LETTERS] if isinstance(opts, dict) else list(opts)
        if len(choices) != 4 or r["answer_idx"] not in LETTERS:
            continue
        out.append([mc_prompt(r["question"], choices), LETTERS.index(r["answer_idx"])])
    return out


def medmcqa_pairs(rows, order_seed=None):
    """[prompt, gold] for MedMCQA rows (dicts with question, opa to opd, cop). Rows without a valid answer or an
    empty option are skipped. With order_seed the rows are shuffled the way skills_eval.medmcqa_items does for 7."""
    idx = list(range(len(rows)))
    if order_seed is not None:
        random.Random(order_seed).shuffle(idx)
    out = []
    for i in idx:
        r = rows[i]
        choices = [r["opa"], r["opb"], r["opc"], r["opd"]]
        cop = r.get("cop")
        if cop is None or int(cop) not in (0, 1, 2, 3) or not str(r["question"]).strip():
            continue
        if any(not str(c).strip() for c in choices):
            continue
        out.append([mc_prompt(r["question"], choices), int(cop)])
    return out


def drop_ngram_overlap(items, index):
    """(kept, dropped): [prompt, gold] items whose prompt shares no 13-gram with the index (wsp.rft)."""
    from wsp import rft
    kept = [it for it in items if not rft.shares_ngram(it[0], index)]
    return kept, len(items) - len(kept)


def check_pool(pool_items, eval_sets):
    """Raise when a pool prompt is an eval prompt. Returns the number of eval prompts checked."""
    banned = set()
    for items in eval_sets.values():
        banned |= {p for p, _ in items}
    bad = [p for p, _ in pool_items if p in banned]
    if bad:
        raise AssertionError("%d pool prompts are eval prompts, first: %r" % (len(bad), bad[0][:80]))
    return len(banned)


def _demo4(question, options):
    lines = ["Question: " + question]
    for i, o in enumerate(options):
        lines.append(LETTERS[i] + ". " + o)
    return "\n".join(lines) + "\nAnswer:"


# Six medical questions written for this demo, two per language, not from any eval set. Gold letters in MED_DEMO_GOLD.
MED_DEMO_PROMPTS = {
    "hi1": _demo4("मानव शरीर में इंसुलिन का उत्पादन कौन सा अंग करता है?", ["यकृत", "अग्न्याशय", "गुर्दा", "प्लीहा"]),
    "hi2": _demo4("रक्त में ऑक्सीजन ले जाने वाला प्रोटीन कौन सा है?", ["कोलेजन", "एल्बुमिन", "हीमोग्लोबिन", "कैरोटीन"]),
    "fr1": _demo4("Quel organe filtre le sang et produit l'urine ?", ["Le foie", "Le rein", "Le poumon", "La rate"]),
    "fr2": _demo4("Quelle vitamine est produite par la peau sous l'effet du soleil ?",
                  ["La vitamine C", "La vitamine K", "La vitamine D", "La vitamine B12"]),
    "ja1": _demo4("心臓の左心室から血液が送り出される血管はどれですか。", ["肺動脈", "大動脈", "大静脈", "門脈"]),
    "ja2": _demo4("インスリンが不足すると血液中で増えるものはどれですか。", ["グルコース", "ヘモグロビン", "カルシウム", "ナトリウム"]),
}
MED_DEMO_GOLD = {"hi1": "B", "hi2": "C", "fr1": "B", "fr2": "C", "ja1": "B", "ja2": "A"}
