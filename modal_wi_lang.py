"""Write-in track language: Arabic, Japanese, Chinese, Korean and Hindi understanding from Qwen2.5-7B-Instruct into
Llama-3.2-1B-Instruct through the parallel graft of docs/write_in_xfam.md. Design and results: docs/write_in_language.md.

Run only on Modal profiles hanavm and rexheng-24542 (owner rule, 2026-11-15).

    MODAL_PROFILE=hanavm modal run modal_wi_lang.py --stage prefetch
    MODAL_PROFILE=hanavm modal run modal_wi_lang.py --stage prep        # CPU: eval sets, pool, texts, report
    MODAL_PROFILE=hanavm modal run modal_wi_lang.py --stage soft        # GPU: donor soft letters for the pool
    MODAL_PROFILE=hanavm modal run --detach modal_wi_lang.py --cells graft_lora:1 --smoke true
    MODAL_PROFILE=hanavm modal run --detach modal_wi_lang.py --cells graft_lora:1,graft_lora:2

Medical track (data version v2med, docs/demo_med_multilingual.md): set WI_LANG_VERSION=v2med before every command.
    WI_LANG_VERSION=v2med MODAL_PROFILE=rexheng-24542 modal run modal_wi_lang.py --stage prefetch_med
    WI_LANG_VERSION=v2med MODAL_PROFILE=rexheng-24542 modal run modal_wi_lang.py --stage prep
    --plan '[["fullft", 0, {"ft_lr": 1e-6, "steps": 1200, "eval": ["dev_mmmlu_hi"], "tag": "_pilot_ft1e-6"}]]'
    --stage verify --cells 1      (rebuild the saved graft of seed 1 from its artifact and re-score)
    --stage laptop --cells 1      (CPU stand-in: 8 cores, 32 GB)

Eval sets (never trained on or selected on): Belebele test in 10 languages, MMMLU held-out questions in five, English
MMLU held-out 2000. Dev slice for learning-rate picks: dev_mmmlu (200 other questions per language).
"""

import json
import os
import random
import time

import modal

VERSION = os.environ.get("WI_LANG_VERSION", "v1")   # v1: the language run; v2med: the medical track (docs/demo_med_multilingual.md)
app = modal.App("wsp-wi-lang" if VERSION == "v1" else "wsp-wi-lang-" + VERSION)
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

def _image(extra_env):
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
            "safetensors", "zstandard", "numpy", "datasets", "pyarrow", "sentencepiece", "protobuf", "hf_xet",
        )
        .env({"HF_HOME": "/vol/hf", "WI_LANG_VERSION": VERSION, **extra_env})
        .add_local_python_source("wsp")
        .add_local_file("modal_graft.py", "/root/modal_graft.py")
        .add_local_file("modal_round5.py", "/root/modal_round5.py")
        .add_local_file("modal_round6.py", "/root/modal_round6.py")
        .add_local_file("modal_round7.py", "/root/modal_round7.py")
        .add_local_file("modal_round22pg.py", "/root/modal_round22pg.py")
        .add_local_file("modal_round23pg.py", "/root/modal_round23pg.py")
    )


online = _image({})
image = _image({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})

MED = VERSION == "v2med"
DIR = f"/vol/wi_lang/{VERSION}"
ROWS = f"{DIR}/rows"
HOST = "unsloth/Llama-3.2-1B-Instruct"
DONOR = "Qwen/Qwen2.5-7B-Instruct"
NATIVES = {"qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
           "llama3b": "unsloth/Llama-3.2-3B-Instruct"}
D_HOST, D_DONOR = 2048, 3584
PRICE = {"H100": 3.95, "H200": 4.54}   # dollars per GPU hour at list price

BASE_CFG = {
    "arm": "graft", "seed": 0, "first": 4, "last": 20, "host_layer": 7,
    "steps": 4000, "batch": 16, "align_steps": 600, "kd_w": 1.0, "preserve": 1.0, "align_reg": 1.0,
    "enc_depth": 2, "dec_rank": 256, "lora_rank": 0, "lora_lr": 2e-4,
    "glue_lr": 3e-4, "gate_lr": 1e-4, "enc_lr": 1e-4, "ft_lr": 3e-6, "match_extra": True,
    "decode": True, "tag": "", "model": None, "eval": None,
}
PRESETS = {
    "graft_lora": {"lora_rank": 16},
    "identity_l": {"arm": "identity", "lora_rank": 16},
    "scrambled_l": {"arm": "scrambled", "lora_rank": 16},
    "lora_match_l": {"arm": "lora_match", "match_lora": 16},
    "lora_big": {"arm": "lora_big", "lora_rank": 128, "lora_lr": 5e-5},
    "fullft": {"arm": "fullft"},
    "reference": {"arm": "reference"},
    "donor": {"arm": "donor"},
    "native": {"arm": "native"},
}
BIG = ("graft", "scrambled")


# ------------------------------------------------------------------ stage 0: downloads (online, CPU)


@app.function(image=online, volumes={"/vol": volume}, cpu=4, memory=16384, timeout=4 * 3600)
def prefetch(models: str = "all"):
    from huggingface_hub import hf_hub_download, snapshot_download

    from wsp import lang_demo as ld

    t0 = time.time()
    report = {}
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    for lang in ld.BELEBELE_ALL:
        hf_hub_download("facebook/belebele", f"data/{lang}.jsonl", repo_type="dataset")
    for f in ld.MMMLU_FILES.values():
        hf_hub_download("openai/MMMLU", f"test/mmlu_{f}.csv", repo_type="dataset")
    hf_hub_download("cais/mmlu", "all/test-00000-of-00001.parquet", repo_type="dataset")
    for lang in ld.XNLI_LANGS:
        hf_hub_download("facebook/xnli", f"{lang}/train-00000-of-00001.parquet", repo_type="dataset")
    for lang in ld.WIKI_LANGS.values():
        hf_hub_download("wikimedia/wikipedia", f"20231101.{lang}/train-00000-of-%05d.parquet" % WIKI_SHARDS[lang],
                        repo_type="dataset")
    print("small data done", round(time.time() - t0))
    for repo in ("cais/mmlu", "tau/commonsense_qa", "allenai/openbookqa", "allenai/qasc", "datatune/LogiQA2.0"):
        try:
            snapshot_download(repo, repo_type="dataset")
        except Exception as e:
            report[repo] = repr(e)[:200]
    print("english pools done", round(time.time() - t0), report)
    wanted = [HOST, DONOR] + list(NATIVES.values()) if models == "all" else models.split(",")
    for m in wanted:
        snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
        print("model", m, round(time.time() - t0))
    volume.commit()
    return {"seconds": round(time.time() - t0), "errors": report}


WIKI_SHARDS = {"en": 41, "ar": 7, "ja": 15, "zh": 6, "ko": 3, "hi": 2, "fr": 17}


# ------------------------------------------------------------------ stage 1: data (offline, CPU)


def _hub(repo, name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name, repo_type="dataset", local_files_only=True)


@app.function(image=image, volumes={"/vol": volume}, cpu=8, memory=65536, timeout=3 * 3600)
def prep():
    import pyarrow.parquet as pq
    import transformers

    from wsp import lang_demo as ld
    from wsp import xfam

    t0 = time.time()
    os.makedirs(DIR, exist_ok=True)
    tok = transformers.AutoTokenizer.from_pretrained(HOST)
    report = {"counts": {}, "alignment": {}}
    evals = {}
    for lang in ld.BELEBELE_ALL:
        items = ld.belebele_items(_hub("facebook/belebele", f"data/{lang}.jsonl"))
        assert len(items) == 900, (lang, len(items))
        evals["bele_" + lang] = items

    mmlu = pq.read_table(_hub("cais/mmlu", "all/test-00000-of-00001.parquet")).to_pylist()
    assert len(mmlu) == ld.N_MMLU_TEST
    split = ld.split_indexes()
    en_prompt = [ld.mc_prompt(r["question"], r["choices"]) for r in mmlu]
    evals["mmlu_held"] = [[en_prompt[i], int(mmlu[i]["answer"])] for i in split["en_held"]]
    en_dev = [[en_prompt[i], int(mmlu[i]["answer"])] for i in split["en_dev"]]

    rows_by_lang, dev = {}, []
    for lang, f in ld.MMMLU_FILES.items():
        rows = ld.read_mmmlu(_hub("openai/MMMLU", f"test/mmlu_{f}.csv"))
        assert len(rows) == ld.N_MMLU_TEST, (lang, len(rows))
        frac = ld.check_alignment(rows, mmlu)
        report["alignment"][lang] = round(frac, 4)
        assert frac > 0.98, ("MMMLU rows are not aligned with the MMLU test order", lang, frac)
        rows_by_lang[lang] = rows
        evals["mmmlu_" + lang] = ld.mmmlu_items(rows, split["held"])
        dev += ld.mmmlu_items(rows, split["dev"])
    evals["dev_mmmlu"] = dev
    report["eval_sizes"] = {k: len(v) for k, v in evals.items()}
    print("evals", report["eval_sizes"], "alignment", report["alignment"])

    banned = set()
    for items in evals.values():
        banned |= {p for p, _ in items}
    banned |= {p for p, _ in en_dev}
    pool, src = [], []
    for lang, rows in rows_by_lang.items():
        items = ld.mmmlu_items(rows, split["train"], mmlu, require_same_answer=True)
        report["counts"]["mmmlu_%s_raw_rows" % lang] = len(split["train"])
        kept, c = ld.filter_items(items, banned, tok, ld.TRAIN_MAX_TOKENS)
        c["answer_mismatch_dropped"] = len(split["train"]) - c["in"]
        report["counts"]["mmmlu_" + lang] = c
        pool += kept
        src += ["mmmlu_" + lang] * len(kept)
    for lang in ld.XNLI_LANGS:
        rows = pq.read_table(_hub("facebook/xnli", f"{lang}/train-00000-of-00001.parquet")).to_pylist()
        items = ld.xnli_items(rows, 4000, seed=11)
        kept, c = ld.filter_items(items, banned, tok, ld.TRAIN_MAX_TOKENS)
        c["source_rows"] = len(rows)
        report["counts"]["xnli_" + lang] = c
        pool += kept
        src += ["xnli_" + lang] * len(kept)
    en_items, n_aux = xfam.build_pool({"banned": [[p, 0] for p in banned]}, aux_count=8000)
    kept, c = ld.filter_items(en_items, banned, tok, ld.TRAIN_MAX_TOKENS)
    c["mmlu_aux_train"] = n_aux
    report["counts"]["english"] = c
    pool += kept
    src += ["english"] * len(kept)
    seen, uniq, uniq_src = set(), [], []
    for item, s in zip(pool, src):
        assert item[0] not in banned, "pool item collides with an eval prompt"
        if item[0] in seen:
            report["counts"].setdefault("cross_source_duplicates_dropped", {}).setdefault(s, 0)
            report["counts"]["cross_source_duplicates_dropped"][s] += 1
            continue
        seen.add(item[0])
        uniq.append(item)
        uniq_src.append(s)
    pool, src = uniq, uniq_src
    order = list(range(len(pool)))
    random.Random(7).shuffle(order)
    pool = [pool[i] for i in order]
    src = [src[i] for i in order]
    report["pool_size"] = len(pool)
    report["pool_by_source"] = {k: src.count(k) for k in sorted(set(src))}
    print("pool", report["pool_size"], report["pool_by_source"])

    # token lengths of the eval prompts with the host tokenizer (truncation report)
    report["eval_token_lengths"] = {}
    for k, items in evals.items():
        lens = ld.token_lengths(tok, [p for p, _ in items])
        report["eval_token_lengths"][k] = {"mean": round(sum(lens) / len(lens), 1), "max": max(lens),
                                           "over_%d" % ld.EVAL_MAX_TOKENS: sum(int(x > ld.EVAL_MAX_TOKENS) for x in lens)}
    print("lengths", json.dumps(report["eval_token_lengths"]))

    # alignment and general text from Wikipedia (first shard of each language)
    texts = {"align": [], "align_eval": [], "general": []}
    for lang in ld.WIKI_LANGS:
        pf = pq.ParquetFile(_hub("wikimedia/wikipedia", f"20231101.{lang}/train-00000-of-%05d.parquet" % WIKI_SHARDS[lang]))
        arts = []
        for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
            arts += batch.column("text").to_pylist()
            if len(arts) >= 20000:
                break
        n_general = 3000 if lang == "en" else 0
        chunks = ld.wiki_chunks(arts, lang, 600 + 12 + n_general, seed=5)
        texts["align"] += chunks[:600]
        texts["align_eval"] += chunks[600:612]
        texts["general"] += chunks[612:]
        report["counts"]["wiki_" + lang] = {"articles_read": len(arts), "chunks": len(chunks)}
    rng = random.Random(0)
    rng.shuffle(texts["align"])
    rng.shuffle(texts["align_eval"])
    rng.shuffle(texts["general"])
    with open(f"{DIR}/evals.json", "w") as f:
        json.dump(evals, f)
    with open(f"{DIR}/pool_nosoft.json", "w") as f:
        json.dump({"items": pool, "src": src}, f)
    with open(f"{DIR}/texts.json", "w") as f:
        json.dump(texts, f)
    with open(f"{DIR}/prep_report.json", "w") as f:
        json.dump(report, f, indent=1)
    volume.commit()
    report["seconds"] = round(time.time() - t0)
    return report


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=3 * 3600)
def soft():
    import torch
    import transformers

    from wsp import lang_demo as ld
    from wsp import xfam

    t0 = time.time()
    volume.reload()
    with open(f"{DIR}/pool_nosoft.json") as f:
        pool = json.load(f)
    items, src = pool["items"], pool["src"]
    dtok = transformers.AutoTokenizer.from_pretrained(DONOR)
    donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()
    probs = xfam.letter_probs(donor, dtok, items, batch=16)
    for i, s in enumerate(src):
        if s.startswith("xnli"):
            probs[i] = ld.restrict_soft(probs[i], 3)
    acc = {}
    for s in sorted(set(src)):
        idx = [i for i, x in enumerate(src) if x == s]
        acc[s] = sum(int(max(range(4), key=lambda j: probs[i][j]) == items[i][1]) for i in idx) / len(idx)
    out = {"items": items, "soft": probs, "src": src, "donor_acc_by_source": acc}
    with open(f"{DIR}/pool.json", "w") as f:
        json.dump(out, f)
    volume.commit()
    print("donor accuracy on the pool by source", acc)
    return {"pool": len(items), "donor_acc_by_source": acc, "seconds": round(time.time() - t0)}


# ------------------------------------------------------------------ stage 1 for v2med: medical track data


@app.function(image=online, volumes={"/vol": volume}, cpu=4, memory=16384, timeout=2 * 3600)
def prefetch_med():
    """Files the medical track needs beyond the language run: French MMMLU, French XNLI and Wikipedia, MedMCQA, MedQA."""
    from huggingface_hub import hf_hub_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from wsp import lang_demo as ld
    for lang in ld.MED_BELEBELE:
        hf_hub_download("facebook/belebele", f"data/{lang}.jsonl", repo_type="dataset")
    for f in ld.MED_LANGS.values():
        hf_hub_download("openai/MMMLU", f"test/mmlu_{f}.csv", repo_type="dataset")
    hf_hub_download("cais/mmlu", "all/test-00000-of-00001.parquet", repo_type="dataset")
    for lang in ld.MED_XNLI:
        hf_hub_download("facebook/xnli", f"{lang}/train-00000-of-00001.parquet", repo_type="dataset")
    for lang in ld.MED_WIKI.values():
        hf_hub_download("wikimedia/wikipedia", f"20231101.{lang}/train-00000-of-%05d.parquet" % WIKI_SHARDS[lang],
                        repo_type="dataset")
    for split in ("train", "validation"):
        hf_hub_download("openlifescienceai/medmcqa", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
    for split in ("train", "test"):
        hf_hub_download("GBaker/MedQA-USMLE-4-options", f"phrases_no_exclude_{split}.jsonl", repo_type="dataset")
    volume.commit()
    return "ok"


@app.function(image=image, volumes={"/vol": volume}, cpu=8, memory=65536, timeout=3 * 3600)
def prep_med():
    import pyarrow.parquet as pq
    import transformers

    from wsp import lang_demo as ld
    from wsp import rft, xfam

    t0 = time.time()
    os.makedirs(DIR, exist_ok=True)
    tok = transformers.AutoTokenizer.from_pretrained(HOST)
    report = {"counts": {}, "alignment": {}, "alignment_med_rows": {}, "answer_mismatch_med_rows": {}}
    evals, dev = {}, {}
    for lang in ld.MED_BELEBELE:
        items = ld.belebele_items(_hub("facebook/belebele", f"data/{lang}.jsonl"))
        assert len(items) == 900, (lang, len(items))
        evals["bele_" + lang] = items

    mmlu = pq.read_table(_hub("cais/mmlu", "all/test-00000-of-00001.parquet")).to_pylist()
    assert len(mmlu) == ld.N_MMLU_TEST
    subjects = [r["subject"] for r in mmlu]
    split = ld.med_split(subjects)
    en_prompt = [ld.mc_prompt(r["question"], r["choices"]) for r in mmlu]
    evals["mmlu_held"] = [[en_prompt[i], int(mmlu[i]["answer"])] for i in split["en_held"]]
    evals["mmlu_med_en"] = [[en_prompt[i], int(mmlu[i]["answer"])] for i in split["med"]]
    en_dev = [[en_prompt[i], int(mmlu[i]["answer"])] for i in split["en_dev"]]
    assert len(evals["mmlu_med_en"]) == len(split["med"])
    report["med_rows_by_subject"] = {s: sum(1 for i in split["med"] if subjects[i] == s) for s in ld.MED6_SUBJECTS}
    report["med_rows_in_en_held"] = len(set(split["med"]) & set(split["en_held"]))

    rows_by_lang = {}
    for lang, f in ld.MED_LANGS.items():
        rows = ld.read_mmmlu(_hub("openai/MMMLU", f"test/mmlu_{f}.csv"))
        assert len(rows) == ld.N_MMLU_TEST, (lang, len(rows))
        frac = ld.check_alignment(rows, mmlu)
        assert frac > 0.98, ("MMMLU rows are not aligned with the MMLU test order", lang, f, frac)
        report["alignment"][lang] = round(frac, 4)
        med_rows = [rows[i] if rows[i] is not None else None for i in split["med"]]
        sub = [mmlu[i] for i in split["med"]]
        report["alignment_med_rows"][lang] = round(ld.check_alignment(med_rows, sub), 4)
        report["answer_mismatch_med_rows"][lang] = sum(
            1 for r, e in zip(med_rows, sub) if r is not None and r["answer"] != int(e["answer"]))
        rows_by_lang[lang] = rows
        evals["mmmlu_med_" + lang] = ld.mmmlu_items(rows, split["med"])
        dev["dev_mmmlu_" + lang] = ld.mmmlu_items(rows, ld.usable_train_rows(rows, subjects, split["dev"]))
    print("alignment", report["alignment"], "on med rows", report["alignment_med_rows"])

    med_val = pq.read_table(_hub("openlifescienceai/medmcqa", "data/validation-00000-of-00001.parquet")).to_pylist()
    order = list(range(len(med_val)))
    random.Random(7).shuffle(order)
    evals["medmcqa_val"] = ld.medmcqa_pairs([med_val[i] for i in order[:1000]])
    assert len(evals["medmcqa_val"]) == 1000, len(evals["medmcqa_val"])
    with open(_hub("GBaker/MedQA-USMLE-4-options", "phrases_no_exclude_test.jsonl"), encoding="utf-8") as f:
        evals["medqa_test"] = ld.jsonl_medqa_items(f.readlines())
    assert len(evals["medqa_test"]) == 1273, len(evals["medqa_test"])

    medmcqa_train = pq.read_table(_hub("openlifescienceai/medmcqa", "data/train-00000-of-00001.parquet")).to_pylist()
    mq = ld.medmcqa_pairs(medmcqa_train, order_seed=11)
    dev["dev_medmcqa"] = mq[:ld.DEV_MED_N]
    mq = mq[ld.DEV_MED_N:]
    with open(_hub("GBaker/MedQA-USMLE-4-options", "phrases_no_exclude_train.jsonl"), encoding="utf-8") as f:
        mqa = ld.jsonl_medqa_items(f.readlines())
    report["eval_sizes"] = {k: len(v) for k, v in evals.items()}
    report["dev_sizes"] = {k: len(v) for k, v in dev.items()}
    print("evals", report["eval_sizes"], "dev", report["dev_sizes"])

    banned = set()
    for items in list(evals.values()) + list(dev.values()) + [en_dev]:
        banned |= {p for p, _ in items}
    index = rft.ngram_index([p for items in list(evals.values()) + list(dev.values()) for p, _ in items])
    pool, src = [], []

    def add(name, items, dedupe_ngram=False, cap=0):
        c = {}
        if dedupe_ngram:
            items, c["dropped_13gram"] = ld.drop_ngram_overlap(items, index)
        kept, c2 = ld.filter_items(items, banned, tok, ld.TRAIN_MAX_TOKENS)
        c.update(c2)
        if cap and len(kept) > cap:
            kept = kept[:cap]
            c["capped_to"] = cap
        report["counts"][name] = c
        pool.extend(kept)
        src.extend([name] * len(kept))

    for lang, rows in rows_by_lang.items():
        idx = ld.usable_train_rows(rows, subjects, split["train"])
        items = ld.mmmlu_items(rows, idx, mmlu, require_same_answer=True)
        add("mmmlu_" + lang, items)
        report["counts"]["mmmlu_" + lang]["candidate_rows"] = len(split["train"])
    for lang in ld.MED_XNLI:
        rows = pq.read_table(_hub("facebook/xnli", f"{lang}/train-00000-of-00001.parquet")).to_pylist()
        add("xnli_" + lang, ld.xnli_items(rows, 4000, seed=11))
    en_items, n_aux = xfam.build_pool({"banned": [[p, 0] for p in banned]}, aux_count=8000)
    add("english", en_items, dedupe_ngram=True)
    report["counts"]["english"]["mmlu_aux_train"] = n_aux
    add("medmcqa_train", mq, dedupe_ngram=True, cap=8000)
    add("medqa_train", mqa, dedupe_ngram=True)

    seen, uniq, uniq_src = set(), [], []
    for item, s in zip(pool, src):
        assert item[0] not in banned, "pool item collides with an eval or dev prompt"
        if item[0] in seen:
            d = report["counts"].setdefault("cross_source_duplicates_dropped", {})
            d[s] = d.get(s, 0) + 1
            continue
        seen.add(item[0])
        uniq.append(item)
        uniq_src.append(s)
    pool, src = uniq, uniq_src
    n_checked = ld.check_pool(pool, {**evals, **dev, "en_dev": en_dev})
    order = list(range(len(pool)))
    random.Random(7).shuffle(order)
    pool = [pool[i] for i in order]
    src = [src[i] for i in order]
    report["pool_size"] = len(pool)
    report["pool_by_source"] = {k: src.count(k) for k in sorted(set(src))}
    report["eval_prompts_checked_against_pool"] = n_checked
    print("pool", report["pool_size"], report["pool_by_source"])

    report["eval_token_lengths"] = {}
    for k, items in evals.items():
        lens = ld.token_lengths(tok, [p for p, _ in items])
        report["eval_token_lengths"][k] = {"mean": round(sum(lens) / len(lens), 1), "max": max(lens),
                                           "over_%d" % ld.EVAL_MAX_TOKENS: sum(int(x > ld.EVAL_MAX_TOKENS) for x in lens)}

    texts = {"align": [], "align_eval": [], "general": []}
    for lang in ld.MED_WIKI:
        pf = pq.ParquetFile(_hub("wikimedia/wikipedia", f"20231101.{lang}/train-00000-of-%05d.parquet" % WIKI_SHARDS[lang]))
        arts = []
        for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
            arts += batch.column("text").to_pylist()
            if len(arts) >= 20000:
                break
        n_general = 3000 if lang == "en" else 0
        chunks = ld.wiki_chunks(arts, lang, 600 + 12 + n_general, seed=5)
        texts["align"] += chunks[:600]
        texts["align_eval"] += chunks[600:612]
        texts["general"] += chunks[612:]
        report["counts"]["wiki_" + lang] = {"articles_read": len(arts), "chunks": len(chunks)}
    rng = random.Random(0)
    for k in texts:
        rng.shuffle(texts[k])
    allsets = {**evals, **dev}
    with open(f"{DIR}/evals.json", "w") as f:
        json.dump(allsets, f)
    with open(f"{DIR}/evals_gold.json", "w") as f:
        json.dump({k: [g for _, g in v] for k, v in allsets.items()}, f)
    with open(f"{DIR}/pool_nosoft.json", "w") as f:
        json.dump({"items": pool, "src": src}, f)
    with open(f"{DIR}/texts.json", "w") as f:
        json.dump(texts, f)
    report["seconds"] = round(time.time() - t0)
    with open(f"{DIR}/prep_report.json", "w") as f:
        json.dump(report, f, indent=1)
    volume.commit()
    return report


# ------------------------------------------------------------------ stage 2: cells


def row_budget(c, config):
    from wsp import pgraft
    from wsp.lora import lora_param_count
    total = (pgraft.count(pgraft.Enc(D_HOST, D_DONOR, depth=c["enc_depth"]).parameters())
             + pgraft.count(pgraft.Dec(D_DONOR, D_HOST, rank=c["dec_rank"]).parameters()) + D_HOST + 1)
    if c.get("match_lora", 0):
        total += lora_param_count(config, c["match_lora"])
    return total


def finish(row, t0, smoke, cname, gpu):
    row["gpu"] = gpu
    row["gpu_hours"] = round((time.time() - t0) / 3600, 3)
    row["dollars_list"] = round(row["gpu_hours"] * PRICE[gpu], 2)
    row["seconds"] = round(time.time() - t0, 1)
    if not smoke:
        os.makedirs(ROWS, exist_ok=True)
        with open(f"{ROWS}/{row['name']}_{row['seed']}{row['config'].get('tag', '')}.json", "w") as f:
            json.dump(row, f)
        volume.commit()
    print(cname, "ROW", json.dumps({k: v for k, v in row.items() if k not in ("demo", "eval_stats", "preds")}))
    return row


def _score(model, tok, evals, budget, preds=None):
    """(scores, stats). With a dict as preds, the per-item predicted option index of every eval set is stored in it."""
    from wsp import lang_demo as ld
    res = ld.score_all(model, tok, evals, budget=budget, return_preds=preds is not None)
    if preds is not None:
        preds.update({k: v["preds"] for k, v in res.items()})
    return {k: v["acc"] for k, v in res.items()}, {k: {x: v[x] for x in ("n", "truncated", "max_len", "mean_len")}
                                                  for k, v in res.items()}


def _demo_prompts(ld):
    return ld.MED_DEMO_PROMPTS if MED else ld.DEMO_PROMPTS


def _demo_new_tokens():
    return 80 if MED else 60


def _cell(name, seed, gpu, smoke=False, overrides=None):
    import gc

    import torch
    import transformers

    from wsp import lang_demo as ld
    from wsp import pgraft, scramble, xfam
    from wsp.lora import apply_lora, default_alpha, rank_for_budget

    cname = f"{name}:{seed}" + (":smoke" if smoke else "")
    t0 = time.time()
    c = {**BASE_CFG, **PRESETS[name], **(overrides or {})}
    c["seed"] = seed
    arm = c["arm"]
    torch.manual_seed(seed)
    try:
        volume.reload()
    except Exception as e:
        print(cname, "reload skipped", repr(e)[:80])
    with open(f"{DIR}/evals.json") as f:
        evals = json.load(f)
    if c["eval"]:
        evals = {k: evals[k] for k in c["eval"]}
    budget = 16384
    row = {"cell": cname, "name": name, "seed": seed, "config": c, "version": VERSION}
    device = torch.device("cuda")

    if arm in ("native", "donor", "reference"):
        model_id = c["model"] or (DONOR if arm == "donor" else HOST)
        if smoke:
            evals = {k: v[:24] for k, v in evals.items()}
        tok = transformers.AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
        row["model"] = model_id
        row["params_per_token"] = row["params_total"] = sum(p.numel() for p in model.parameters())
        row["trainable"] = 0
        row["tokens_trained"] = 0
        row["preds"] = {}
        row["scores"], row["eval_stats"] = _score(model, tok, evals, budget, row["preds"] if MED else None)
        print(cname, "SCORES", json.dumps(row["scores"]))
        row["demo"] = ld.demo_outputs(model, tok, _demo_prompts(ld), new_tokens=_demo_new_tokens())
        if not smoke:
            row["decode_tps"] = xfam.decode_tps(model, tok)
        return finish(row, t0, smoke, cname, gpu)

    with open(f"{DIR}/pool.json") as f:
        pool = json.load(f)
    with open(f"{DIR}/texts.json") as f:
        data = json.load(f)
    items, soft_labels = pool["items"], pool["soft"]
    if smoke:
        c["steps"], c["align_steps"] = 3, 3
        items, soft_labels = items[:64], soft_labels[:64]
        evals = {k: v[:24] for k, v in evals.items()}
        data = {k: v[:16] for k, v in data.items()}
    row["pool"] = len(items)
    cfg = dict(pgraft.DEFAULTS, align_steps=c["align_steps"], geo_steps=100)
    tok = transformers.AutoTokenizer.from_pretrained(HOST)
    letters = xfam.letters_of(tok)
    dtype = torch.float32 if arm == "fullft" else torch.bfloat16
    base = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=dtype).to("cuda").eval()
    for p in base.parameters():
        p.requires_grad_(arm == "fullft")
    ref = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    row["params_base"] = sum(p.numel() for p in base.parameters())

    donor = donor_tok = None
    graft = None
    grafts = []
    lora_params = []
    needs_donor = arm in ("graft", "scrambled")
    if needs_donor:
        donor_tok = transformers.AutoTokenizer.from_pretrained(DONOR)
        donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()
        for p in donor.parameters():
            p.requires_grad_(False)

    joint_lora = c["lora_rank"] if arm in ("graft", "identity", "scrambled") else 0
    if joint_lora:
        lora_params = apply_lora(base, joint_lora, default_alpha(joint_lora))
        for p in lora_params:
            p.requires_grad_(True)

    if arm in ("graft", "identity", "scrambled"):
        control = "identity" if arm == "identity" else None
        g0 = pgraft.build(base, donor if control is None else None, c["host_layer"], c["first"], c["last"],
                          control=control, enc_depth=c["enc_depth"], dec_rank=c["dec_rank"], d_donor=D_DONOR)
        grafts = [g0]
        pgraft.attach(base, grafts)
        graft = g0
        if needs_donor:
            pgraft.set_sink(g0, donor, donor_tok, g0.donor_span[0], data["align"][:256])
        glue = pgraft.trainable(grafts, enc=True)
        row["trainable_glue"] = pgraft.count(glue)
        row["slice_params"] = sum(p.numel() for p in g0.part.parameters()) if g0.control is None else 0
        if needs_donor:
            pgraft.align(base, tok, donor, donor_tok, g0, data["align"], cfg, seed=seed, log_every=100)
            row["alignment"] = pgraft.alignment_report(base, tok, donor, donor_tok, g0, data["align_eval"], cfg)
            print(cname, "alignment", row["alignment"])
        row["calib_rms"] = pgraft.calibrate_scale(base, tok, g0, data["align"], cfg)
        if arm == "scrambled":
            import copy
            g0.part = copy.deepcopy(g0.part)
            rep = scramble.scramble_module(g0.part, seed=seed + 100, mode="spectrum")
            row["scramble_cos"] = rep.get("mean_abs_cosine")
            del donor
            donor = None
            gc.collect()
            torch.cuda.empty_cache()
        trainable_total = pgraft.count(glue) + pgraft.count(lora_params)
    elif arm in ("lora_match", "lora_big"):
        rank = rank_for_budget(base, row_budget(c, base.config)) if arm == "lora_match" else c["lora_rank"]
        lora_params = apply_lora(base, rank, default_alpha(rank))
        for p in lora_params:
            p.requires_grad_(True)
        row["lora_rank"] = rank
        trainable_total = pgraft.count(lora_params)
    else:
        trainable_total = sum(p.numel() for p in base.parameters())
    row["trainable"] = trainable_total

    if arm in ("graft", "identity", "scrambled"):
        gate = [p for g in grafts for p in g.token_gate.parameters()]
        dec = [p for g in grafts for p in g.dec.parameters()]
        enc = [p for g in grafts for p in g.enc.parameters()]
        groups = [{"params": dec + (enc if arm == "identity" else []), "lr": c["glue_lr"]},
                  {"params": gate, "lr": c["gate_lr"]}]
        if arm != "identity":
            groups.append({"params": enc, "lr": c["enc_lr"]})
        if lora_params:
            groups.append({"params": lora_params, "lr": c["lora_lr"]})
    elif arm == "fullft":
        groups = [{"params": list(base.parameters()), "lr": c["ft_lr"]}]
    else:
        groups = [{"params": lora_params, "lr": c["lora_lr"]}]

    steps = c["steps"]
    if arm != "graft" and c["match_extra"]:
        steps += c["align_steps"]
    row["train_steps"] = steps

    rng = random.Random(seed + 5)
    order = list(range(len(items)))
    rng.shuffle(order)
    ptr = [0]

    def next_idx():
        if ptr[0] + c["batch"] > len(order):
            rng.shuffle(order)
            ptr[0] = 0
        idx = order[ptr[0]:ptr[0] + c["batch"]]
        ptr[0] += c["batch"]
        return idx

    gen_feed = pgraft._batches(data["general"], 4, seed + 41)
    align_feed = pgraft._batches(data["align"], cfg["align_batch"], seed + 11) if arm == "graft" else None
    tokens = [0]

    def step_fn(step):
        idx = next_idx()
        batch = [tuple(items[i]) for i in idx]
        sp = torch.tensor([soft_labels[i] for i in idx])
        loss, info = pgraft.mc_soft_loss(base, tok, batch, letters, device, soft=sp, kd_weight=c["kd_w"])
        tokens[0] += info.pop("tokens")
        if c["preserve"] > 0:
            enc = pgraft._tok_batch(tok, next(gen_feed), cfg["seq_len"], device)
            _, logits = pgraft.seq_lm_loss(base, enc, return_logits=True)
            drift = pgraft.drift_kl(ref, enc, logits)
            loss = loss + c["preserve"] * drift
            info["drift"] = float(drift.detach())
        if arm == "graft" and c["align_reg"] > 0 and step % 2 == 0:
            for g in grafts:
                rl, _ = pgraft.functional_loss(base, tok, donor, donor_tok, g, next(align_feed), cfg)
                loss = loss + c["align_reg"] * rl / len(grafts)
        if graft is not None:
            gm = pgraft.mean_gate(grafts)
            if gm is not None:
                info["gate"] = float(gm.detach())
        return loss, info

    log = pgraft.train(base, groups, steps, step_fn, seed=seed + 1, log_every=100)
    row["tokens_trained"] = tokens[0]
    row["gate_curve"] = [[r["step"], round(r["gate"], 4)] for r in log if "gate" in r and r["step"] % 100 == 0]
    row["loss_first50"] = sum(r["loss"] for r in log[:50]) / max(1, len(log[:50]))
    row["loss_last50"] = sum(r["loss"] for r in log[-50:]) / max(1, len(log[-50:]))
    print(cname, "train done", "loss", round(row["loss_first50"], 4), "->", round(row["loss_last50"], 4))
    row["train_seconds"] = round(time.time() - t0, 1)
    row["trained_on"] = ("pool.json: MMMLU hi, fr, ja outside every MED6 row, the v1 held-out and dev indexes and the xfam "
                         "English indexes; XNLI train hi, fr; English xfam pool; MedMCQA train and MedQA train after "
                         "13-gram decontamination; no Belebele item and no eval or dev prompt" if MED else
                         "pool.json: MMMLU outside held-out, dev and the xfam English indexes; XNLI train ar, zh, hi; "
                         "English xfam pool; no Belebele item and no eval or dev prompt")

    if arm == "fullft":
        base = base.to(torch.bfloat16)
    base.eval()
    row["preds"] = {}
    row["scores"], row["eval_stats"] = _score(base, tok, evals, budget, row["preds"] if MED else None)
    print(cname, "SCORES", json.dumps(row["scores"]))
    if grafts and arm != "identity":
        with pgraft.grafts_off(base):
            off = {k: evals[k] for k in ("mmlu_held", "mmmlu_ar", "bele_arb_Arab", "mmmlu_med_hi", "bele_hin_Deva")
                   if k in evals}
            row["scores_graft_off"], _ = _score(base, tok, off, budget)
        print(cname, "SCORES graft off", json.dumps(row["scores_graft_off"]))
    per_token = row["params_base"]
    if grafts and arm != "identity":
        per_token += row["slice_params"] + row["trainable_glue"]
    elif grafts:
        per_token += row["trainable_glue"]
    row["params_per_token"] = row["params_total"] = per_token
    row["demo"] = ld.demo_outputs(base, tok, _demo_prompts(ld), new_tokens=_demo_new_tokens())
    if c.get("save") and arm == "graft" and not smoke:
        row["artifact"] = _save_artifact(base, graft, c, row, seed, tok)
    if c["decode"] and not smoke:
        row["decode_tps"] = xfam.decode_tps(base, tok)
    return finish(row, t0, smoke, cname, gpu)


ART_DIR = f"{DIR}/artifacts"   # one folder per saved graft seed; inside it xfam writes pgraft_v1 (format version in the path)


def _save_artifact(base, graft, c, row, seed, tok):
    from wsp import xfam
    spec = {"lora_rank": c["lora_rank"], "host_layer": c["host_layer"], "first": c["first"], "last": c["last"],
            "enc_depth": c["enc_depth"], "dec_rank": c["dec_rank"], "d_donor": D_DONOR, "d_host": D_HOST}
    meta = {"host": HOST, "donor": DONOR, "data_version": VERSION, "seed": seed, "scores": row["scores"],
            "params_per_token": None, "note": "rebuild with modal_wi_lang.py --stage verify or wsp.xfam.load_artifact"}
    path = xfam.save_artifact(f"{ART_DIR}/graft_lora_{seed}", base, xfam.Built([graft], [], {}), spec, meta)
    volume.commit()
    return path


def _proc_kb(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    return -1


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=2 * 3600)
def verify(seed: int):
    """Rebuild the saved graft from its artifact (no training) and score two eval sets; compare with the cell's row."""
    import torch
    import transformers

    from wsp import lang_demo as ld
    from wsp import xfam
    volume.reload()
    path = f"{ART_DIR}/graft_lora_{seed}/{xfam.ARTIFACT_DIR}"
    model, meta = xfam.load_artifact(path, device="cuda")
    tok = transformers.AutoTokenizer.from_pretrained(HOST)
    with open(f"{DIR}/evals.json") as f:
        evals = json.load(f)
    names = ["mmmlu_med_hi", "mmmlu_med_fr", "mmmlu_med_ja", "dev_mmmlu_hi"]
    got = {k: v["acc"] for k, v in ld.score_all(model, tok, {k: evals[k] for k in names}, budget=16384).items()}
    with open(f"{ROWS}/graft_lora_{seed}_final.json") as f:
        row = json.load(f)
    out = {"seed": seed, "rebuilt": got, "row": {k: row["scores"][k] for k in names},
           "max_abs_diff": max(abs(got[k] - row["scores"][k]) for k in names)}
    print("VERIFY", json.dumps(out))
    os.makedirs(f"{DIR}/verify", exist_ok=True)
    with open(f"{DIR}/verify/graft_lora_{seed}.json", "w") as f:
        json.dump(out, f)
    volume.commit()
    return out


@app.function(image=image, volumes={"/vol": volume}, cpu=8, memory=32768, timeout=3 * 3600)
def laptop(seed: int, new_tokens: int = 32):
    """CPU stand-in for a laptop: 8 cores, 32 GB. Decode speed at batch 1 and peak memory for base and saved graft."""
    import gc

    import torch
    import transformers

    from wsp import lang_demo as ld
    from wsp import xfam
    torch.set_num_threads(8)
    volume.reload()
    tok = transformers.AutoTokenizer.from_pretrained(HOST)
    prompts = {"en": "The patient has a fever and a cough. The doctor should",
               "fr": "Le patient a de la fievre et une toux. Le medecin doit",
               "hi": "मरीज को बुखार और खांसी है। डॉक्टर को"}

    def decode(model):
        res = {}
        for name, text in prompts.items():
            enc = tok(text, return_tensors="pt")
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=3, do_sample=False)
                t = time.time()
                out = model.generate(**enc, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False)
                dt = time.time() - t
            res[name] = {"tokens": int(out.shape[1] - enc["input_ids"].shape[1]), "seconds": round(dt, 2),
                         "tokens_per_second": round((out.shape[1] - enc["input_ids"].shape[1]) / dt, 3)}
        return res

    result = {"cpus": os.cpu_count(), "threads": torch.get_num_threads(), "dtype": "bfloat16", "new_tokens": new_tokens,
              "cpu_model": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?"),
              "bf16_flags": [w for w in open("/proc/cpuinfo").read().split() if w.startswith("avx512_bf16") or w == "amx_bf16"][:2]}
    try:
        base = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).eval()
        result["base"] = {"params": sum(p.numel() for p in base.parameters()), "decode": decode(base),
                          "rss_gb": round(_proc_kb("VmRSS") / 1e6, 2), "peak_gb": round(_proc_kb("VmHWM") / 1e6, 2)}
        print("LAPTOP base", json.dumps(result["base"]))
        del base
        gc.collect()
        try:
            with open("/proc/self/clear_refs", "w") as f:
                f.write("5")
        except Exception:
            pass
    except Exception as e:
        result["base_error"] = repr(e)[:400]
    try:
        t = time.time()
        model, meta = xfam.load_artifact(f"{ART_DIR}/graft_lora_{seed}/{xfam.ARTIFACT_DIR}", device="cpu", dtype=torch.bfloat16)
        result["graft_load_seconds"] = round(time.time() - t, 1)
        result["graft_peak_gb_during_load"] = round(_proc_kb("VmHWM") / 1e6, 2)
        result["graft_rss_gb_after_load"] = round(_proc_kb("VmRSS") / 1e6, 2)
        print("LAPTOP graft loaded", json.dumps({k: v for k, v in result.items() if k.startswith("graft")}))
        result["graft_decode"] = decode(model)
        result["graft_peak_gb"] = round(_proc_kb("VmHWM") / 1e6, 2)
        print("LAPTOP graft decode", json.dumps(result["graft_decode"]))
        result["graft_demo"] = ld.demo_outputs(model, tok, ld.MED_DEMO_PROMPTS, new_tokens=80)
    except Exception as e:
        import traceback
        result["graft_error"] = repr(e)[:400]
        print("LAPTOP graft failed", traceback.format_exc()[-1500:])
    os.makedirs(f"{DIR}/laptop", exist_ok=True)
    with open(f"{DIR}/laptop/standin_seed{seed}.json", "w") as f:
        json.dump(result, f)
    volume.commit()
    return result


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=5 * 3600)
def cell(name, seed, smoke=False, overrides=None):
    return _cell(name, seed, "H100", smoke, overrides)


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=131072, timeout=6 * 3600)
def cell_big(name, seed, smoke=False, overrides=None):
    return _cell(name, seed, "H200", smoke, overrides)


@app.local_entrypoint()
def main(stage: str = "cells", cells: str = "", smoke: bool = False, overrides: str = "", models: str = "all", plan: str = ""):
    if stage == "prefetch":
        print(json.dumps(prefetch.remote(models)))
        return
    if stage == "prep":
        print(json.dumps(prep_med.remote() if MED else prep.remote()))
        return
    if stage == "prefetch_med":
        print(json.dumps(prefetch_med.remote()))
        return
    if stage == "verify":
        print(json.dumps(verify.remote(int(cells or 1))))
        return
    if stage == "laptop":
        print(json.dumps(laptop.remote(int(cells or 1))))
        return
    if stage == "soft":
        print(json.dumps(soft.remote()))
        return
    calls = []
    for item in [x for x in cells.split(",") if x]:
        name, seed = item.split(":")
        if name not in PRESETS:
            raise SystemExit(f"unknown preset {name}")
        calls.append((name, int(seed), smoke, json.loads(overrides) if overrides else None))
    for name, seed, ov in (json.loads(plan) if plan else []):
        if name not in PRESETS:
            raise SystemExit(f"unknown preset {name}")
        calls.append((name, int(seed), smoke, ov))
    if not calls:
        raise SystemExit("0 cells: pass --cells name:seed,... or --plan '[[name, seed, {overrides}], ...]'")
    handles = []
    for x in calls:
        arm = {**BASE_CFG, **PRESETS[x[0]], **(x[3] or {})}["arm"]
        handles.append((cell_big if arm in BIG else cell).spawn(*x))
    for h in handles:
        try:
            h.get()
        except Exception as e:
            print("cell failed:", repr(e)[:1000])
