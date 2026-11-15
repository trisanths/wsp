"""Round 27: identity-seam skill tops. Fine-tuned members need no learned stitch.

Question. A true fine-tune of the host has the host's shapes, tokenizer ids,
and positional base, and its weights moved little. So the host's state at
layer k is already in the member's coordinates. The network keeps the host's
layers 0..cut-1 as a shared trunk and adds, per member, the member's layers
from the cut, its norm, and its head (wsp/tops_identity.py). The seam is the
identity, nothing trains, and a closed-form ridge router (lambda 0.01) picks
the host's own top or a member's top per sequence from the trunk's mean state
at the cut. This is Experiment 2 of docs/research/2026-11-15-transfer-sweep.md.

Host Qwen/Qwen2.5-7B-Instruct. Members, both fine-tunes of the host:
  code       nvidia/OpenCodeReasoning-Nemotron-7B
  reasoning  open-thoughts/OpenThinker-7B
Cuts 7, 14, 21 (1/4, 1/2, 3/4 of 28 layers).

Cells per cut (names start with R27_):
  forced code           the code member's top on the trunk. mbpp, humaneval, mmlu
  forced reasoning      the reasoning member's top. gsm8k, mmlu
  cross code on math    the code top on gsm8k
  cross reasoning code  the reasoning top on mbpp and humaneval
  routed                both tops and the host's own top behind the router. All four
  scrambled (seeds 0,1) the code top with every weight matrix rewritten by wsp.scramble.
                        mbpp, humaneval, mmlu. Short token limit
References (no cut): the host, each member alone, Qwen/Qwen2.5-Coder-7B-Instruct.
All four benchmarks.

Scoring. MBPP 257 (all sanitized test items), HumanEval 164, GSM8K 300, MMLU
1000 (the round 6 dev items). Greedy decoding, so there is no sampling seed;
the scrambled control has two scramble seeds. Generative cells that run a
reasoning member or top get 4096 new tokens, the host and the scrambled top get
the usual 512 (wsp.skills_eval.MAX_NEW_TOKENS). The limit is in every row and on
every score line. Two raw generations per benchmark print after each score.
The tops and the host use the host's chat template; a member alone uses its
own tokenizer and template.

Rows record stored params and params run per path. A row is written to the
volume after every benchmark (partial) and when the cell ends, so a lost
client loses nothing. A finished row on the volume is reused.

No benchmark data trains anything (owner rule, 2026-11-14). Nothing here
trains. The router fit, its probe, and the base margin read only wsp.healdata
pools under the default reported set (healdata.check_sources runs on every
pool), in three disjoint ranges. Scoring items come from wsp.skills_eval.

Usage.
    modal run --detach modal_round27.py --smoke
    modal run --detach modal_round27.py --nshards 3 --shard 0 --only forced,cross
    modal run modal_round27.py --plan-only
    python modal_round27.py sizes
Run the smoke first and wait for its first score line. A fan-out comes only after it.
"""

import json
import os
import time

import modal

app = modal.App("wsp-round27")
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

HOST = "Qwen/Qwen2.5-7B-Instruct"
MEMBERS = {"code": "nvidia/OpenCodeReasoning-Nemotron-7B", "reasoning": "open-thoughts/OpenThinker-7B"}
CODER = "Qwen/Qwen2.5-Coder-7B-Instruct"
N_LAYERS = 28
CUTS = [7, 14, 21]
ALL_SKILLS = ["mbpp", "humaneval", "gsm8k", "mmlu"]
OWN_SKILLS = {"code": ["mbpp", "humaneval"], "reasoning": ["gsm8k"]}
COUNTS = {"gsm8k": 300, "mmlu": 1000}
EVAL_BATCH = 32
SHORT_TOKENS = {"gsm8k": 512, "humaneval": 512, "mbpp": 512}
LONG_TOKENS = 4096
SCRAMBLE_SEEDS = [0, 1]
RAW_PRINTED = 2
GO_ROUTER_ACCURACY = 0.95

# Three disjoint ranges of every healdata pool: router fit, router probe. Pools read round robin in a fixed order.
RANGES = {"router": (1200, 400), "probe": (1600, 150)}
ROUTER = {"code": [("code", 300)], "reasoning": [("math", 300)],
          "base": [("general", 150), ("knowledge", 100), ("reasoning", 100), ("medicine", 100)]}
ROUTER_PROBE = {"code": [("code", 100)], "reasoning": [("math", 100)],
                "base": [("general", 40), ("knowledge", 30), ("reasoning", 30), ("medicine", 30)]}
ROUTER_LAMBDA = 0.01
ROUTER_SCALE = 8.0

DATA_DIR = "/vol/round27/data"
RESULTS_DIR = "/vol/round27/results"
FETCH_DIR = "/vol/round27/fetch"
GPU_SPEC = {"gpu": "H200", "memory": 131072, "timeout": 6 * 3600}
MAX_CONTAINERS = 8


# ------------------------------------------------------------------ cells


def new_cell(kind, top=None, cut=None, seed=0, skills=None, long=True, **extra):
    cell = {"kind": kind, "top": top, "cut": cut, "seed": seed, "skills": list(skills), "long": bool(long)}
    cell.update(extra)
    return cell


def cell_name(cell):
    parts = ["R27", cell["kind"], cell["top"] or "none"]
    if cell["cut"] is not None:
        parts.append("c" + str(cell["cut"]))
    parts.append("s" + str(cell["seed"]))
    if cell.get("smoke"):
        parts.append("smoke")
    return "_".join(parts)


def cells():
    """Every cell: references first, then the cuts in order."""
    out = [new_cell("host", skills=ALL_SKILLS, long=False)]
    for top in MEMBERS:
        out.append(new_cell("member", top=top, skills=ALL_SKILLS))
    out.append(new_cell("coder", skills=ALL_SKILLS, long=False))
    for cut in CUTS:
        out.append(new_cell("forced", top="code", cut=cut, skills=OWN_SKILLS["code"] + ["mmlu"]))
        out.append(new_cell("forced", top="reasoning", cut=cut, skills=OWN_SKILLS["reasoning"] + ["mmlu"]))
        out.append(new_cell("cross", top="code", cut=cut, skills=OWN_SKILLS["reasoning"]))
        out.append(new_cell("cross", top="reasoning", cut=cut, skills=OWN_SKILLS["code"]))
        out.append(new_cell("routed", top="all", cut=cut, skills=ALL_SKILLS))
        for seed in SCRAMBLE_SEEDS:
            out.append(new_cell("scrambled", top="code", cut=cut, seed=seed, skills=OWN_SKILLS["code"] + ["mmlu"],
                                long=False))
    return out


def smoke_cells():
    """Forced code top at cut 14 on MBPP, against the code member alone and the host."""
    out = [new_cell("forced", top="code", cut=14, skills=["mbpp"]),
           new_cell("member", top="code", skills=["mbpp"]),
           new_cell("host", skills=["mbpp"], long=False)]
    for cell in out:
        cell["smoke"] = True
    return out


def select(todo, only="", cuts=""):
    out = []
    wanted = [int(c) for c in cuts.split(",") if c.strip()] if cuts else []
    kinds = [k for k in only.split(",") if k] if only else []
    for cell in todo:
        if kinds and cell["kind"] not in kinds:
            continue
        if wanted and cell["cut"] is not None and cell["cut"] not in wanted:
            continue
        out.append(cell)
    return out


def token_limits(cell):
    """New tokens per generative skill for a cell."""
    if cell["long"]:
        return {skill: LONG_TOKENS for skill in SHORT_TOKENS}
    return dict(SHORT_TOKENS)


# ------------------------------------------------------------------ router data


def safe_name(text):
    return text.replace("/", "__")


def pool_prompts(base_tokenizer, pool, role, count, suffix):
    """User prompts of a healdata pool in one range, in the chat format and the raw format the MMLU items use."""
    from wsp import healdata
    from wsp import skills_eval as ev

    skip, size = RANGES[role]
    if count > size:
        raise ValueError(f"role {role} holds at most {size} texts per pool, asked for {count}")
    path = os.path.join(DATA_DIR, f"prompts_{role}_{pool}_{count}_{suffix}_{safe_name(HOST)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    healdata.check_sources([source[0] for source in healdata.SOURCES[pool]])
    pairs = healdata.pairs(pool, count, seed=0, skip=skip)
    if len(pairs) < count:
        raise ValueError(f"pool {pool} gave {len(pairs)} prompts for {role}, wanted {count}")
    prompts = []
    for index, (user, _) in enumerate(pairs):
        text = user
        if suffix == "reasoning":
            text = user + "\n" + (ev.GSM8K_INSTRUCTION if index % 2 == 0 else ev.MATH_INSTRUCTION)
        elif suffix == "code":
            text = user + "\n\n" + ev.CODE_INSTRUCTION
        prompts.append(ev.render_prompt(base_tokenizer, text))
        prompts.append("Question: " + user.strip() + "\nAnswer:")
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(prompts, f)
    # No hard links on a Modal volume: a plain rename. A file that is already there is kept.
    if os.path.exists(path):
        os.remove(tmp)
    else:
        os.replace(tmp, path)
    return prompts


def router_texts(base_tokenizer, tops, table, role):
    out = {}
    for name in ["base"] + list(tops):
        texts = []
        for pool, count in table[name]:
            suffix = name if name in ("code", "reasoning") else "general"
            texts += pool_prompts(base_tokenizer, pool, role, count, suffix)
        out[name] = texts
    return out


# ------------------------------------------------------------------ scoring helpers


def load_items(skills):
    from wsp import skills_eval as ev

    items = {}
    if "mbpp" in skills:
        items["mbpp"] = ev.mbpp_items()
    if "humaneval" in skills:
        items["humaneval"] = ev.humaneval_items()
    if "gsm8k" in skills:
        items["gsm8k"] = ev.gsm8k_items(COUNTS["gsm8k"])
    if "mmlu" in skills:
        items["mmlu"] = ev.mmlu_items(COUNTS["mmlu"])
    return items


def decode_speed(model, tokenizer, new_tokens=128, repeats=2):
    """Greedy decode tokens per second at batch 1. Never stops on EOS, so every run makes new_tokens."""
    import torch

    from wsp import skills_eval as ev

    ev.prepare_tokenizer(tokenizer)
    text = ev.render_prompt(tokenizer, "Explain how a binary search works, step by step.")
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    never = [len(tokenizer) + 5000]
    best = 0.0
    for run in range(repeats + 1):
        count = 8 if run == 0 else new_tokens
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=count, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=never)
        torch.cuda.synchronize()
        seconds = time.time() - start
        made = out.shape[1] - ids.shape[1]
        if run > 0 and seconds > 0 and made / seconds > best:
            best = made / seconds
    return best


def write_row(row, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name + ".json")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(row, f, default=str)
    os.replace(tmp, path)
    volume.commit()


def score_model(name, model, tokenizer, cell, items, row, tally_owner=None):
    """Score every skill of a cell. One score line per skill, then two raw generations. Rows go to the volume per skill."""
    from wsp import skills_eval as ev

    limits = token_limits(cell)
    row.setdefault("scores", {})
    row.setdefault("counts", {})
    row.setdefault("samples", {})
    row.setdefault("route_shares", {})
    row["max_new_tokens"] = {skill: limits[skill] for skill in cell["skills"] if skill in limits}
    for skill in cell["skills"]:
        if tally_owner is not None:
            tally_owner.tally = {}
        result = ev.evaluate_skill(model, tokenizer, skill, items=items[skill], batch_size=EVAL_BATCH,
                                   max_new_tokens=limits.get(skill))
        row["scores"][skill] = result["score"]
        row["counts"][skill] = {"correct": result["correct"], "n": result["n"]}
        limit = limits.get(skill, "logits")
        line = (f"{name} score {skill} {result['score']:.4f} ({result['correct']}/{result['n']}) "
                f"limit {limit} in {result['seconds']} s")
        if tally_owner is not None:
            total = sum(tally_owner.tally.values())
            row["route_shares"][skill] = {k: v / max(1, total) for k, v in tally_owner.tally.items()}
            line += " route " + json.dumps({k: round(v, 3) for k, v in row["route_shares"][skill].items()})
            tally_owner.tally = None
        print(line, flush=True)
        samples = []
        for record in result["records"][:RAW_PRINTED]:
            if "output" in record:
                samples.append({"id": record["id"], "correct": record["correct"], "output": record["output"]})
        row["samples"][skill] = samples
        for index, sample in enumerate(samples):
            print(f"{name} raw {skill} {index} id {sample['id']} correct {sample['correct']}: "
                  f"{json.dumps(sample['output'])}", flush=True)
        row["partial"] = True
        write_row(row, name)


# ------------------------------------------------------------------ building


def load_model(model_id):
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    return model, tokenizer


def row_base(cell):
    return {"cell": cell_name(cell), "kind": cell["kind"], "top": cell["top"], "cut": cell["cut"],
            "seed": cell["seed"], "skills": cell["skills"], "long": cell["long"], "smoke": bool(cell.get("smoke")),
            "host": HOST, "members": MEMBERS}


def lineage_of(member_id):
    """The lineage relation of a member against the host. A definite wrong kind refuses; a read error is recorded."""
    from wsp import lineage
    from wsp.tops_identity import TRUE_FINE_TUNE

    try:
        found = lineage.relation(HOST, member_id)
    except Exception as error:
        return {"kind": None, "error": repr(error)[:300]}
    kind = found.get("kind")
    if kind not in TRUE_FINE_TUNE:
        raise ValueError(f"{member_id} is {kind!r} against {HOST}, not a true fine-tune; the identity seam stops")
    return {"kind": kind, "warnings": found.get("warnings", [])}


def finish_net_row(row, net):
    counts = net.count_params()
    row["params"] = counts
    row["params_stored"] = counts["stored"]
    row["added_over_base"] = counts["added_over_base"]
    row["stored_over_base"] = counts["stored_over_base"]
    row["params_run_per_path"] = counts["run"]


def finish(row, start, name):
    row["seconds"] = round(time.time() - start, 1)
    row["partial"] = False
    write_row(row, name)
    print(f"{name} done in {row['seconds']} s scores {json.dumps(row.get('scores'))}", flush=True)
    return row


def run_cell(cell):
    import torch

    from wsp import skills_eval as ev
    from wsp.tops import ForcedView
    from wsp.tops_identity import IdentityTops, fit_identity_router, route_accuracy

    start = time.time()
    safe_reload()
    torch.manual_seed(cell["seed"])
    name = cell_name(cell)
    path = os.path.join(RESULTS_DIR, name + ".json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if not old.get("partial"):
            print(f"{name} reused finished row", flush=True)
            return old
    kind = cell["kind"]
    row = row_base(cell)
    items = load_items(cell["skills"])
    print(f"{name} start kind {kind} skills {cell['skills']} long {cell['long']}", flush=True)

    if kind in ("host", "member", "coder"):
        model_id = {"host": HOST, "coder": CODER}.get(kind) or MEMBERS[cell["top"]]
        model, tokenizer = load_model(model_id)
        row["model"] = model_id
        score_model(name, model, tokenizer, cell, items, row)
        row["tokens_per_second"] = decode_speed(model, tokenizer)
        row["params_total"], row["params_run"] = ev.model_params(model)
        row["params_stored"] = row["params_total"]
        row["params_run_per_path"] = {"alone": row["params_run"]}
        return finish(row, start, name)

    base, tokenizer = load_model(HOST)
    if len(base.model.layers) != N_LAYERS:
        raise ValueError(f"{HOST} has {len(base.model.layers)} layers, the plan assumed {N_LAYERS}")
    net = IdentityTops(base, tokenizer, base_id=HOST)
    names = list(MEMBERS) if kind == "routed" else [cell["top"]]
    row["lineage"] = {}
    for top in names:
        row["lineage"][top] = lineage_of(MEMBERS[top])
        control = "scrambled" if kind == "scrambled" else None
        built = time.time()
        net.add_top(top, MEMBERS[top], cell["cut"], control=control, control_seed=cell["seed"], device="cuda",
                    cache_dir=FETCH_DIR, lineage=row["lineage"][top].get("kind"))
        print(f"{name} built top {top} from {MEMBERS[top]} at cut {cell['cut']} in {time.time() - built:.1f} s "
              f"lineage {json.dumps(row['lineage'][top])} vocab {json.dumps(net.tops[top].spec['vocab'])[:300]}",
              flush=True)
    finish_net_row(row, net)
    print(f"{name} stored {row['params_stored']} ({row['stored_over_base']:.3f}x host) run per path "
          f"{json.dumps(row['params_run_per_path'])}", flush=True)

    if kind == "routed":
        texts = router_texts(tokenizer, names, ROUTER, "router")
        probe = router_texts(tokenizer, names, ROUTER_PROBE, "probe")
        report = fit_identity_router(net, texts, lam=ROUTER_LAMBDA, scale=ROUTER_SCALE, probe_by_class=probe,
                                     max_length=384, batch_size=8)
        accuracy = route_accuracy(net, probe, max_length=384, batch_size=8)
        row["router"] = {"fit": report, "probe_accuracy": accuracy, "base_margin": net.base_margin,
                         "go_accuracy": GO_ROUTER_ACCURACY, "go": accuracy["overall"] >= GO_ROUTER_ACCURACY}
        print(f"{name} router classes {report['classes']} cut {report['cut']} lam {ROUTER_LAMBDA} margin "
              f"{net.base_margin} probe accuracy {json.dumps({k: round(v, 3) for k, v in accuracy.items()})}",
              flush=True)
        finish_net_row(row, net)
        score_model(name, net, tokenizer, cell, items, row, tally_owner=net)
        model = net
    else:
        model = ForcedView(net, cell["top"])
        score_model(name, model, tokenizer, cell, items, row)
    row["tokens_per_second"] = decode_speed(model, tokenizer)
    return finish(row, start, name)


@app.function(image=image, volumes={"/vol": volume}, gpu=GPU_SPEC["gpu"], memory=GPU_SPEC["memory"],
              timeout=GPU_SPEC["timeout"], max_containers=MAX_CONTAINERS)
def cell(spec):
    return run_cell(spec)


@app.function(image=image, volumes={"/vol": volume}, memory=16384, timeout=4 * 3600)
def prefetch(donor_id, first_layer):
    """Fetch a member's layers first_layer..end, final norm, and head by byte range into the volume cache."""
    import transformers

    from wsp.parts import block_tensor_names
    from wsp.skill_moe import fetch_expert_tensors

    safe_reload()
    config = transformers.AutoConfig.from_pretrained(donor_id)
    n = config.num_hidden_layers
    names = block_tensor_names(config, first_layer, n - 1)
    head = "model.embed_tokens.weight" if getattr(config, "tie_word_embeddings", False) else "lm_head.weight"
    os.makedirs(FETCH_DIR, exist_ok=True)
    fetch_expert_tensors(donor_id, names + ["model.norm.weight", head], FETCH_DIR)
    volume.commit()
    return {"donor": donor_id, "tensors": len(names) + 2}


# ------------------------------------------------------------------ sizes and plan


def real_config(repo):
    """The model config from the lineage facts kept in library/lineage/facts, else from the Hub (metadata only)."""
    import transformers

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library", "lineage", "facts",
                        repo.replace("/", "__") + ".json")
    if os.path.exists(path):
        with open(path) as f:
            facts = json.load(f)["config"]
        return transformers.Qwen2Config(
            hidden_size=facts["hidden"], intermediate_size=facts["mlp_size"], num_hidden_layers=facts["layers"],
            num_attention_heads=facts["heads"], num_key_value_heads=facts["kv_heads"], vocab_size=facts["vocab"],
            rope_theta=facts["rope_theta"], max_position_embeddings=facts["max_positions"],
            tie_word_embeddings=bool(facts["tied_embeddings"]))
    return transformers.AutoConfig.from_pretrained(repo)


def sizes_table():
    """Stored and per-token params per cut, from the real configs (no weights)."""
    from wsp.tops_identity import identity_sizes

    host = real_config(HOST)
    table = {}
    for cut in CUTS:
        tops = [{"name": top, "donor_config": real_config(repo), "cut_host": cut} for top, repo in MEMBERS.items()]
        sizes = identity_sizes(host, tops, router_classes=1 + len(tops))
        single = identity_sizes(host, tops[:1], router_classes=None)
        table[cut] = sizes
        run = {k: round(v / 1e9, 3) for k, v in sizes["run"].items()}
        print(f"sizes cut {cut}: host {sizes['base_total'] / 1e9:.3f}B routed network stored "
              f"{sizes['stored'] / 1e9:.3f}B ({sizes['stored_over_base']:.3f}x) one forced top stored "
              f"{single['stored'] / 1e9:.3f}B ({single['stored_over_base']:.3f}x) run per path (B) {run}")
    return table


def plan(todo=None):
    todo = cells() if todo is None else todo
    for item in todo:
        print(f"plan {cell_name(item)} skills {item['skills']} long {item['long']}")
    print(f"plan {len(todo)} cells, at most {MAX_CONTAINERS} containers at once")


# ------------------------------------------------------------------ summary


def summarize(rows):
    by_name = {row["cell"]: row for row in rows}
    print(f"{'cell':44s} " + " ".join(f"{s:>9s}" for s in ALL_SKILLS) + "   stored(B)")
    for name in sorted(by_name):
        row = by_name[name]
        scores = row.get("scores") or {}
        cols = [f"{scores[s]:9.4f}" if s in scores else f"{'':>9s}" for s in ALL_SKILLS]
        stored = row.get("params_stored")
        print(f"{name:44s} " + " ".join(cols) + (f"   {stored / 1e9:.3f}" if stored else ""))
        shares = row.get("route_shares")
        if shares:
            print(f"{name} route shares {json.dumps(shares)} router {json.dumps((row.get('router') or {}).get('probe_accuracy'))}")


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, only: str = "", cuts: str = "", smoke: bool = False,
         plan_only: bool = False, no_prefetch: bool = False, summary: str = ""):
    if summary:
        with open(summary) as f:
            summarize(json.load(f))
        return
    todo = smoke_cells() if smoke else select(cells(), only, cuts)
    mine = [spec for index, spec in enumerate(todo) if index % nshards == shard]
    print(f"shard {shard}/{nshards}: {len(mine)} cells")
    plan(mine)
    if plan_only:
        return
    if not no_prefetch:
        wanted = {}
        for spec in mine:
            if spec["kind"] in ("host", "member", "coder"):
                continue
            tops = list(MEMBERS) if spec["kind"] == "routed" else [spec["top"]]
            for top in tops:
                wanted[MEMBERS[top]] = min(spec["cut"], wanted.get(MEMBERS[top], spec["cut"]))
        for found in prefetch.starmap(list(wanted.items())):
            print("prefetched", found, flush=True)
    rows = []
    out_path = "results/round27_smoke.json" if smoke else f"results/round27_shard{shard}.json"
    os.makedirs("results", exist_ok=True)
    for row in cell.map(mine, order_outputs=False, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            continue
        rows.append(row)
        print(f"{row['cell']} row scores {json.dumps(row.get('scores'))} stored {row.get('params_stored')} "
              f"tps {row.get('tokens_per_second')}", flush=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)
    summarize(rows)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sizes":
        sizes_table()
    elif len(sys.argv) > 1 and sys.argv[1] == "plan":
        plan()
