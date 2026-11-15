"""Round 16d: skill deltas. One host network, one mined low-rank delta per specialist, one expert per sequence.

Run 16b showed that a specialist keeps its skill only when its weights run
as it was trained (Qwen2.5-Math-1.5B-Instruct 81.7 on GSM8K, the base 68.7,
a per-token router over MLP-only experts stayed at the base). Here each
specialist is mined as D = W_specialist - W_host per matrix, kept as a
rank-r factor pair (wsp/delta.py), and a ridge router on the host's own
early hidden state picks one delta per prompt. The expert is fixed for the
whole sequence. Params run per token are the host plus the active factors.
Stored params are the host plus every routed delta plus the router. Nothing
trains except the ridge router (a closed-form fit).
No benchmark data trains anything: router texts come from wsp.healdata pools only.

Pair A: host Qwen/Qwen2.5-1.5B-Instruct, experts Qwen/Qwen2.5-Math-1.5B-Instruct
        and Qwen/Qwen2.5-Coder-1.5B-Instruct.
Pair B: host Qwen/Qwen2.5-7B-Instruct, experts Qwen/Qwen2.5-Math-7B-Instruct,
        Qwen/Qwen2.5-Coder-7B-Instruct, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B.
        The Hugging Face config.json and tokenizer.json of R1-Distill were
        compared with the host on 2026-11-14: same layers, hidden size,
        heads, MLP width, vocab_size 152064, the same 151643-token BPE vocab
        and merges. Only the names of seven special tokens (ids 151643 to
        151649) differ, and rope_theta is 10000 (host 1e6). So it is mined.
        If a mining run finds a shape mismatch, wsp.delta skips that matrix
        and reports it in "skipped". R1 has no clean non-benchmark domain
        text, so it is NOT in the routed bank and not a router class. It
        keeps its alone, own template, forced, and control cells.

Scores (wsp.skills_eval, identical prompts and items for every arm, the host
tokenizer and chat template): GSM8K 300, MATH-500 200 (seeded), MBPP 200
(seeded), HumanEval 164, MMLU 1000, MedMCQA 1000. The two "own template"
cells run a specialist with its own tokenizer, as in run 16b, for reference.

Cells per pair:
  host                         the host alone
  spec_<e>                     each specialist alone, host template
  spec_own_<e>                 each specialist alone, its own template (own skills and MMLU)
  routed_r<R>, R in full, 256, 64, 16
                               the routed network; the bank is Math and Coder
                               (R1 is left out of the bank and the router)
  forced_<e>_r<R>, R in 64, full
                               expert e forced on every prompt (own skills and MMLU);
                               separates router errors from compression loss
  control_<e>_r64              random delta, same shapes and per-matrix norms as the
                               rank 64 delta of e, forced (own skills and MMLU);
                               carries no norms, biases, or rope change
Rows hold every score, router choice shares per benchmark, stored params,
params per token, decode tokens per second per expert, and the per-matrix
delta rank statistics (rank for 90% and 99% energy, error, energy kept).

Mining: rank 256 SVD factors per expert (embeddings and head included) are
cached in /vol/delta16d once; lower ranks are cut from them; "full" cells
subtract the weights on the fly (bf16 factors on the GPU). The router is fit
once per pair on domain texts from wsp.healdata pools (seed 0): "math" for
Math, "code" for Coder, "general", "prose", and "medicine" for the base (a
third each). Layer 4, ridge 1.0, base_margin 0.15, 10% held out for the
accuracy in the row. Every text that shares a 13-word shingle or a 100
character prefix with any scoring item is dropped; the counts are in the row,
with the pool names and dataset ids used.

Go rule: at some rank with stored params at most 1.3x the host, the routed
network is within 2 points of each specialist on its own skill and within 1
point of the host on MMLU, and the random control is at or below the host.

    python scripts/modal_fanout.py modal_round16d.py round16d --profiles a,b,c

Writes results/round16d_shard<i>.json and full per-item records to
/vol/results/round16d/<pair>_<cell>.json.
"""

import json
import os
import random
import re
import time

import modal

app = modal.App("wsp-round16d")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)


def safe_reload():
    """Reload the volume to see other shards' files. Open files (HF logs, caches) block a reload; then skip it."""
    try:
        volume.reload()
    except Exception as error:
        print("volume reload skipped:", repr(error)[:200])

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)

PAIRS = {
    "a": {
        "host": "Qwen/Qwen2.5-1.5B-Instruct",
        "experts": {"math": "Qwen/Qwen2.5-Math-1.5B-Instruct", "coder": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
    },
    "b": {
        "host": "Qwen/Qwen2.5-7B-Instruct",
        "experts": {"math": "Qwen/Qwen2.5-Math-7B-Instruct", "coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
    },
}
OWN_SKILLS = {"math": ["gsm8k", "math500"], "coder": ["mbpp", "humaneval"], "r1": ["gsm8k", "math500"]}
ALL_SKILLS = ["gsm8k", "math500", "humaneval", "mbpp", "mmlu", "medmcqa"]
COUNTS = {"gsm8k": 300, "math500": 200, "mbpp": 200, "mmlu": 1000, "medmcqa": 1000}
RANKS = ["full", 256, 64, 16]
MINE_RANK = 256
FORCED_RANKS = [64, "full"]
CONTROL_RANK = 64

ROUTER_LAYER = 4
ROUTER_RIDGE = 1.0
BASE_MARGIN = 0.15
ROUTER_HOLDOUT = 0.1
PER_CLASS = 600
ROUTED = ["math", "coder"]
# Which healdata pools feed each router class. Nothing here is a scored benchmark.
TRAIN_POOLS = {"math": ["math"], "coder": ["code"], "base": ["general", "prose", "medicine"]}
SEED = 0
CACHE = "/vol/delta16d"


def rank_name(rank):
    return "full" if rank == "full" else str(rank)


def cells():
    out = []
    for pair in ["a", "b"]:
        experts = list(PAIRS[pair]["experts"])
        out.append((pair, "host"))
        for e in experts:
            out.append((pair, "spec_" + e))
        for e in experts:
            out.append((pair, "spec_own_" + e))
        for rank in RANKS:
            out.append((pair, "routed_r" + rank_name(rank)))
        for e in experts:
            for rank in FORCED_RANKS:
                out.append((pair, f"forced_{e}_r{rank_name(rank)}"))
        for e in experts:
            out.append((pair, f"control_{e}_r{CONTROL_RANK}"))
    return out


# ---------------------------------------------------------------- items


def seeded_pick(items, count, seed=0):
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    picked = []
    for index in sorted(order[:count]):
        picked.append(items[index])
    return picked


def eval_items(ev):
    items = {
        "gsm8k": ev.gsm8k_items(count=COUNTS["gsm8k"], seed=0),
        "math500": seeded_pick(ev.math500_items(), COUNTS["math500"]),
        "humaneval": ev.humaneval_items(),
        "mbpp": seeded_pick(ev.mbpp_items(), COUNTS["mbpp"]),
        "mmlu": ev.mmlu_items(count=COUNTS["mmlu"]),
        "medmcqa": ev.medmcqa_items(count=COUNTS["medmcqa"]),
    }
    return items


def route_text(skill, item):
    """The raw content the router reads: no chat template, no instruction suffix."""
    if skill == "gsm8k":
        return item["question"]
    if skill == "math500":
        return item["problem"]
    if skill == "humaneval":
        return item["prompt"]
    if skill == "mbpp":
        return item["text"] + "\n" + "\n".join(item["test_list"])
    return item["prompt"]


# ---------------------------------------------------------------- router data


def shingles(text, size=13):
    words = re.findall(r"\w+", text.lower())
    out = set()
    if len(words) < size:
        out.add(" ".join(words))
        return out
    for index in range(len(words) - size + 1):
        out.add(" ".join(words[index:index + size]))
    return out


def drop_overlaps(texts, scoring_texts):
    """Drop every text that shares a 13-word shingle or a 100 character prefix with a scoring text."""
    banned = set()
    prefixes = set()
    for text in scoring_texts:
        banned |= shingles(text)
        prefixes.add(text.strip()[:100].lower())
    kept = []
    dropped = 0
    for text in texts:
        if text.strip()[:100].lower() in prefixes or len(shingles(text) & banned) > 0:
            dropped += 1
        else:
            kept.append(text)
    return kept, dropped


def routed_experts(pair):
    """Experts in the routed bank of a pair. R1 is not routed."""
    return [e for e in PAIRS[pair]["experts"] if e in ROUTED]


def source_ids():
    """Dataset ids of every pool the router uses. Raises when one is a reported benchmark."""
    from wsp import healdata

    ids = []
    for names in TRAIN_POOLS.values():
        for name in names:
            for source in healdata.SOURCES[name]:
                ids.append(source[0])
    return healdata.check_sources(ids)


def class_texts(klass, count):
    """Router texts (the user side only) of one class, split evenly over its healdata pools."""
    from wsp import healdata

    names = TRAIN_POOLS[klass]
    share = (count + len(names) - 1) // len(names)
    texts = []
    for name in names:
        for user, _ in healdata.pairs(name, share + 200, seed=SEED, skip=0):
            texts.append(user)
    return texts


def router_texts(pair, items):
    """Return (texts_by_expert, notes). Pools only. No text overlaps any scoring item."""
    scoring = []
    for skill in items:
        for item in items[skill]:
            scoring.append(route_text(skill, item))
    notes = {"dropped": {}, "counts": {}, "missing": [], "sources": source_ids(), "pools": TRAIN_POOLS,
             "classes": routed_experts(pair) + ["base"]}
    out = {}
    for name in routed_experts(pair) + ["base"]:
        kept, dropped = drop_overlaps(class_texts(name, PER_CLASS), scoring)
        out[name] = kept[:PER_CLASS]
        notes["dropped"][name] = dropped
        notes["counts"][name] = len(out[name])
    return out, notes


# ---------------------------------------------------------------- models and mining


def load_model(name):
    import torch
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()


def cache_path(pair, expert):
    return f"{CACHE}/{pair}/{expert}_r{MINE_RANK}.pt"


def mined_delta(pair, expert):
    import torch

    safe_reload()
    return torch.load(cache_path(pair, expert), weights_only=False)


def delta_at_rank(pair, expert, rank):
    """The delta of one expert at a rank: cut from the cache, or a dense one made on the fly."""
    import torch

    from wsp.delta import mine_delta

    if rank == "full":
        return mine_delta(PAIRS[pair]["host"], PAIRS[pair]["experts"][expert], "full", name=expert,
                          include_embeddings=True, device="cuda", factor_dtype=torch.bfloat16, spectrum=False,
                          keep_spectrum=False)
    delta = mined_delta(pair, expert)
    if rank == MINE_RANK:
        return delta
    return delta.truncate(rank)


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=262144, timeout=6 * 3600)
def prepare(pair):
    """Mine rank 256 deltas for every expert of a pair, then fit and cache the router."""
    import torch
    import transformers

    from wsp.delta import SkillDeltaNetwork, mine_delta
    from wsp import skills_eval as ev

    start = time.time()
    spec = PAIRS[pair]
    os.makedirs(f"{CACHE}/{pair}", exist_ok=True)
    safe_reload()
    for expert in spec["experts"]:
        path = cache_path(pair, expert)
        if os.path.exists(path):
            continue
        delta = mine_delta(spec["host"], spec["experts"][expert], MINE_RANK, name=expert, include_embeddings=True,
                           device="cuda")
        torch.save(delta, path + ".tmp")
        os.replace(path + ".tmp", path)
        volume.commit()
        print(expert, "mined", round(time.time() - start), "s")
    router_file = f"{CACHE}/{pair}/router.pt"
    if not os.path.exists(router_file):
        host = load_model(spec["host"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(spec["host"])
        deltas = []
        for expert in routed_experts(pair):
            deltas.append(mined_delta(pair, expert).truncate(16))
        net = SkillDeltaNetwork(host, deltas, tokenizer=tokenizer, host_id=spec["host"])
        texts, notes = router_texts(pair, eval_items(ev))
        report = net.fit_router(texts, layer=ROUTER_LAYER, ridge=ROUTER_RIDGE, base_margin=BASE_MARGIN,
                                holdout=ROUTER_HOLDOUT, seed=SEED)
        report["data"] = notes
        torch.save({"router": net.router, "report": report}, router_file + ".tmp")
        os.replace(router_file + ".tmp", router_file)
        volume.commit()
        print("router", json.dumps(report)[:600])
    return {"pair": pair, "seconds": round(time.time() - start, 1)}


# ---------------------------------------------------------------- evaluation


def evaluate_network(net, tokenizer, ev, items, skills, choice):
    """Score skills with one expert per item. choice[skill][i] is the expert name of item i."""
    scores = {}
    records_by = {}
    for skill in skills:
        groups = {}
        for index, name in enumerate(choice[skill]):
            groups.setdefault(name, []).append(index)
        records = [None] * len(items[skill])
        for name in groups:
            net.set_expert(name)
            result = ev.evaluate_skill(net.model, tokenizer, skill, items=[items[skill][i] for i in groups[name]])
            for index, record in zip(groups[name], result["records"]):
                records[index] = record
        correct = 0
        for record in records:
            if record["correct"]:
                correct += 1
        scores[skill] = correct / max(1, len(records))
        records_by[skill] = records
        print(f"{skill}: {scores[skill]:.4f} ({correct}/{len(records)})")
    net.set_expert("base")
    return scores, records_by


def shares_of(choice, names):
    out = {}
    for skill in choice:
        total = max(1, len(choice[skill]))
        out[skill] = {}
        for name in names:
            out[skill][name] = round(sum(1 for c in choice[skill] if c == name) / total, 4)
    return out


def skills_for(cell, experts):
    if cell == "host" or cell.startswith("spec_") and not cell.startswith("spec_own_") or cell.startswith("routed_"):
        return list(ALL_SKILLS)
    name = cell.split("_")[2] if cell.startswith("spec_own_") else cell.split("_")[1]
    return OWN_SKILLS[name] + ["mmlu"]


def parse_rank(text):
    return "full" if text == "full" else int(text)


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=6 * 3600)
def run_cell_small(pair, cell):
    return run_cell(pair, cell)


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=8 * 3600)
def run_cell_big(pair, cell):
    return run_cell(pair, cell)


def run_cell(pair, cell):
    import torch
    import transformers

    from wsp.delta import SkillDeltaNetwork, host_param_counts, random_control, summarize_stats
    from wsp import skills_eval as ev

    start = time.time()
    torch.manual_seed(SEED)
    spec = PAIRS[pair]
    experts = list(spec["experts"])
    host_id = spec["host"]
    row = {"pair": pair, "cell": cell, "host": host_id, "experts": spec["experts"], "ranks_mined": MINE_RANK}
    tokenizer = transformers.AutoTokenizer.from_pretrained(host_id)
    items = eval_items(ev)
    skills = skills_for(cell, experts)
    row["skills"] = skills
    first = items["gsm8k"][0]
    net = None
    choice = None
    if cell == "host" or cell.startswith("spec_"):
        if cell == "host":
            model = load_model(host_id)
        elif cell.startswith("spec_own_"):
            name = cell[len("spec_own_"):]
            model = load_model(spec["experts"][name])
            tokenizer = transformers.AutoTokenizer.from_pretrained(spec["experts"][name])
        else:
            model = load_model(spec["experts"][cell[len("spec_"):]])
        row["gsm8k_prompt_sample"] = ev.render_prompt(tokenizer, ev.user_text("gsm8k", first))
        result = ev.evaluate_suite(model, tokenizer, skills=skills, items=items)
        scores = dict(result["scores"])
        records = {}
        for skill in result["skills"]:
            records[skill] = result["skills"][skill]["records"]
        total, run = host_param_counts(model)
        row["params_stored"] = total
        row["params_per_token"] = run
        row["decode_tok_s"] = {"alone": round(ev.decode_speed(model, tokenizer), 1)}
    else:
        host = load_model(host_id)
        if cell.startswith("routed_"):
            rank = parse_rank(cell[len("routed_r"):])
            bank = routed_experts(pair)
            deltas = []
            for e in bank:
                deltas.append(delta_at_rank(pair, e, rank))
            forced = None
        elif cell.startswith("forced_"):
            _, e, rtext = cell.split("_")
            rank = parse_rank(rtext[1:])
            bank = [e]
            deltas = [delta_at_rank(pair, e, rank)]
            forced = e
        else:
            _, e, rtext = cell.split("_")
            rank = parse_rank(rtext[1:])
            bank = [e]
            deltas = [random_control(delta_at_rank(pair, e, MINE_RANK), rank=rank, seed=SEED, name=e)]
            forced = e
        row["rank"] = rank
        row["forced"] = forced
        net = SkillDeltaNetwork(host, deltas, tokenizer=tokenizer, host_id=host_id)
        row["gsm8k_prompt_sample"] = ev.render_prompt(tokenizer, ev.user_text("gsm8k", first))
        choice = {}
        if forced is None:
            safe_reload()
            saved = torch.load(f"{CACHE}/{pair}/router.pt", weights_only=False)
            net.router = saved["router"]
            row["router_report"] = saved["report"]
            for skill in skills:
                texts = []
                for item in items[skill]:
                    texts.append(route_text(skill, item))
                choice[skill] = net.route_many(texts, batch_size=16)
            row["route_share"] = shares_of(choice, net.expert_names())
            print(json.dumps(row["route_share"]))
        else:
            for skill in skills:
                choice[skill] = [forced] * len(items[skill])
        scores, records = evaluate_network(net, tokenizer, ev, items, skills, choice)
        counts = net.count_params()
        row["params_host"] = counts["host"]
        row["params_stored"] = counts["stored"]
        row["params_deltas"] = counts["deltas"]
        row["params_router"] = counts["router"]
        row["stored_over_host"] = round(counts["stored"] / counts["host"], 4)
        row["params_per_token_by_expert"] = {}
        row["decode_tok_s"] = {}
        for name in net.expert_names():
            row["params_per_token_by_expert"][name] = net.count_params(name)["per_token"]
            net.set_expert(name)
            row["decode_tok_s"][name] = round(ev.decode_speed(net.model, tokenizer), 1)
        net.set_expert("base")
        row["params_per_token"] = counts["per_token"]
        row["delta_stats"] = {}
        row["delta_attr_differences"] = {}
        for e in bank:
            source = mined_delta(pair, e)
            row["delta_stats"][e] = summarize_stats(source.stats)
            row["delta_attr_differences"][e] = source.attrs.get("differences")
            row["delta_attr_skipped"] = source.attrs.get("skipped")
            row["embedding_rel_diff"] = source.attrs.get("embedding_rel_diff_if_excluded")
    row["scores"] = scores
    total = 0.0
    for skill in scores:
        total += scores[skill]
    row["macro"] = total / max(1, len(scores))
    row["seconds"] = round(time.time() - start, 1)
    os.makedirs("/vol/results/round16d", exist_ok=True)
    full = dict(row)
    full["records"] = records
    if choice is not None:
        full["choice"] = choice
    with open(f"/vol/results/round16d/{pair}_{cell}.json", "w") as f:
        json.dump(full, f, default=str)
    volume.commit()
    return row


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1):
    mine = []
    for index, cell in enumerate(cells()):
        if index % nshards == shard:
            mine.append(cell)
    print(f"shard {shard}/{nshards}: {mine}")
    pairs = []
    for pair, _ in mine:
        if pair not in pairs:
            pairs.append(pair)
    for pair in pairs:
        print("prepare", pair, prepare.remote(pair))
    small = [c for c in mine if c[0] == "a"]
    big = [c for c in mine if c[0] == "b"]
    rows = []
    results = list(run_cell_small.starmap(small, return_exceptions=True)) if small else []
    results += list(run_cell_big.starmap(big, return_exceptions=True)) if big else []
    for row in results:
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            continue
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ["pair", "cell", "scores", "params_stored", "params_per_token"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round16d_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2)
