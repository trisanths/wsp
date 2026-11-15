"""Round 20: the synthesis compiler (wsp/synth.py) against LoRA, distillation, and a random control.

Question. Does compiling mined components into one network, with learned
translations at every seam and a short heal, beat the base, base plus LoRA
on the same data, base plus distillation at the same budget, and the same
compile with random same-size parts? Nothing but the one network runs at
inference.

Pairs (all donors share the base tokenizer, so the heal uses donor KL):
  p1  bigger same-family donor. Base Qwen/Qwen2.5-0.5B-Instruct. Component:
      a block of Qwen/Qwen2.5-7B-Instruct, layers 8..20, placed at relative
      depth (host layer 7). Variant synth_top: the same block, exit at host
      layer 21 (two host layers left to read it). Target skills: knowledge
      (MMLU held-out 2000, the order[1000:3000] items of modal_round7.mmlu_split;
      MedMCQA 1000). Guard: GSM8K 300.
  p2  same-size specialists. Base Qwen/Qwen2.5-1.5B-Instruct. Components: a
      block of Qwen/Qwen2.5-Math-1.5B-Instruct (layers 8..17) and a block of
      Qwen/Qwen2.5-Coder-1.5B-Instruct (layers 14..23). Targets: GSM8K 300,
      MBPP 200. Guard: MMLU 1000 (the round 6 dev items).
  p3  library parts. Base Qwen/Qwen2.5-1.5B-Instruct. Components: every
      Qwen/Qwen2.5-Math-1.5B attention-heads and MLP-cluster row of
      library/catalog.jsonl (mined by another agent), each at its own layer.
      Target: GSM8K 300. Guard: MMLU 1000.

Arms per pair (2 seeds for every trained arm, seeds 0 and 1):
  base           the base alone
  kd_lora        LoRA r16 on every layer, donor KL on the same texts, same
                 steps (stage 1 + stage 2), same preservation term and schedule
  lora           the same with the LM loss only (no donor logits)
  synth          real components, both stages (also compiles its random
                 control inside, so the report holds content gain and verdicts)
  synth_stage1   translations, gates only (stage2_steps 0)
  synth_random   random same-size parts of the same donor (control_seed = seed), both stages
  synth_old_out  zero T_out init, LM loss only, no function matching (what the old output side did)
  synth_top      p1 only: exit near the host top

Heal data rule (owner, 2026-11-14): no benchmark data trains anything. Nothing that trains or fits anything
uses a reported benchmark in any split. Heal texts are prompts from the
wsp.healdata pools (math: orca-math; code: Magicoder and CodeAlpaca;
medicine: medalpaca; general: UltraChat and alpaca; prose: wikitext) with the
DONOR's own greedy answers (domain text) or the BASE's own greedy answers
(general text, the preservation term). healdata.check_sources runs on every
dataset id. Practice sets feed only the compile-time guard and the component
test; they never receive gradient and share no item with the scoring sets:
GSM8K train 100, MBPP sanitized train and validation 100, MMLU validation 150,
MedMCQA train 150 (built in wsp/practice.py, which is exempt like wsp.skills_eval). Scoring sets are the test or validation items used in
rounds 6, 7, 10, and 16b.

Rows. results/round20_shard<i>.json: per cell the pair, arm, seed, every
score (fractions), the practice scores, the full compile report (seam
fidelity, gate means per text set, guard events, verdicts, content gain),
params total and per token, and heal seconds.

Go rule. synth beats kd_lora and synth_random by more than the seed spread
on the pair's target skill, with the guard skill within 1 point of the base.

    python scripts/modal_fanout.py modal_round20.py round20 --profiles a,b,c
    modal run modal_round20.py --shard 0 --nshards 3
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-round20")
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

MATH_BASE = "Qwen/Qwen2.5-Math-1.5B"
MATH_INSTRUCT = "Qwen/Qwen2.5-Math-1.5B-Instruct"
CODER_INSTRUCT = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BIG = "Qwen/Qwen2.5-7B-Instruct"
SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
MID = "Qwen/Qwen2.5-1.5B-Instruct"

SEEDS = [0, 1]
TRAINED = ["kd_lora", "lora", "synth", "synth_stage1", "synth_random", "synth_old_out"]
GEN_TOKENS = 256
GSM8K_INSTRUCTION = "Solve step by step and end with: The answer is N."
CODE_INSTRUCTION = "Write the whole function, with its imports, in a single ```python code block."

HEAL = {"stage1_steps": 300, "stage2_steps": 300, "batch_size": 4, "seq_len": 512, "lora_rank": 16,
        "steps_in": 100, "steps_lm": 60, "batch_in": 4, "tolerance": 0.04, "margin": 0.02,
        "lr_lora": 5e-5}

PAIRS = {
    "p1": {
        "base": SMALL,
        "skills": ["mmlu", "medmcqa", "gsm8k"], "targets": ["mmlu", "medmcqa"], "guard": ["gsm8k"],
        "arms": TRAINED + ["synth_top"],
        # key: (generator model, [(pool, count)])
        "domain": {"big": (BIG, [("general", 300), ("medicine", 300)])},
        "preserve": [("general", 200), ("math", 200)],
    },
    "p2": {
        "base": MID,
        "skills": ["gsm8k", "mbpp", "mmlu"], "targets": ["gsm8k", "mbpp"], "guard": ["mmlu"],
        "arms": TRAINED,
        "domain": {"math": (MATH_INSTRUCT, [("math", 600)]), "coder": (CODER_INSTRUCT, [("code", 600)])},
        "preserve": [("general", 200), ("medicine", 100)],
    },
    "p3": {
        "base": MID,
        "skills": ["gsm8k", "mmlu"], "targets": ["gsm8k"], "guard": ["mmlu"],
        "arms": TRAINED,
        "domain": {"math": (MATH_INSTRUCT, [("math", 600)])},
        "preserve": [("general", 200), ("medicine", 100)],
    },
}


def cells():
    out = []
    for pair, spec in PAIRS.items():
        out.append((pair, "base", 0))
        for arm in spec["arms"]:
            for seed in SEEDS:
                out.append((pair, arm, seed))
    return out


# ------------------------------------------------------------------ data


def pool_prompts(pool, count, seed=0, skip=0):
    """User prompts of one healdata pool, in the scorer's prompt format. Never benchmark data."""
    from wsp import healdata

    healdata.check_sources([source[0] for source in healdata.SOURCES[pool]])
    prompts = []
    for user, _ in healdata.pairs(pool, count, seed=seed, skip=skip):
        if pool == "math":
            prompts.append(user + "\n" + GSM8K_INSTRUCTION)
        elif pool == "code":
            prompts.append(user + "\n\n" + CODE_INSTRUCTION)
        else:
            prompts.append(user)
    return prompts


def answers_of(model, tokenizer, prompts):
    """Greedy answers of one model. Items whose answer hit the token limit are dropped."""
    from wsp import skills_eval as ev

    rendered = [ev.render_prompt(tokenizer, p) for p in prompts]
    outputs = ev.generate_texts(model, tokenizer, rendered, max_new_tokens=GEN_TOKENS, batch_size=64)
    pairs = []
    for prompt, answer in zip(prompts, outputs):
        answer = answer.strip()
        if len(answer) == 0:
            continue
        if len(tokenizer(answer, add_special_tokens=False)["input_ids"]) >= GEN_TOKENS - 2:
            continue
        pairs.append({"prompt": prompt, "answer": answer})
    return pairs


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=131072, timeout=4 * 3600)
def prep(pair):
    """Build and cache the heal texts of one pair (donor answers for domain text, base answers for general text)."""
    import torch
    import transformers

    from wsp import healdata

    path = f"/vol/round20/data_{pair}.json"
    os.makedirs("/vol/round20", exist_ok=True)
    safe_reload()
    if os.path.exists(path):
        return path
    spec = PAIRS[pair]
    data = {"domain": {}, "general": [], "align": []}
    for key, (gen_id, pools) in spec["domain"].items():
        tok = transformers.AutoTokenizer.from_pretrained(gen_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(gen_id, dtype=torch.bfloat16).to("cuda").eval()
        prompts = []
        for pool, count in pools:
            prompts += pool_prompts(pool, count, seed=0)
        data["domain"][key] = answers_of(model, tok, prompts)
        del model
        torch.cuda.empty_cache()
    base_tok = transformers.AutoTokenizer.from_pretrained(spec["base"])
    base = transformers.AutoModelForCausalLM.from_pretrained(spec["base"], dtype=torch.bfloat16).to("cuda").eval()
    prompts = []
    for pool, count in spec["preserve"]:
        prompts += pool_prompts(pool, count, seed=1)
    data["general"] = answers_of(base, base_tok, prompts)
    data["align"] = healdata.pool("prose", 150, tokenizer=base_tok, seed=3)
    for item in data["general"][:100]:
        data["align"].append(healdata.render(base_tok, item["prompt"], item["answer"]))
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    try:
        os.link(tmp, path)  # the first writer wins; a later shard keeps the file it finds
    except FileExistsError:
        pass
    os.remove(tmp)
    volume.commit()
    return path


# ------------------------------------------------------- scoring and practice sets


# ------------------------------------------------------------ components


def library_rows(path="library/catalog.jsonl"):
    """Every screened Qwen2.5-Math-1.5B attention-heads and MLP-cluster row, with its unit indices read here (local side).

    Provisional rows were selected by ablation on GSM8K items. That is benchmark use and the p3 target is GSM8K,
    so only rows with tier "screened" (skill screens such as fraction_role) are taken.
    """
    import torch

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["model"] != MATH_BASE or row["type"] not in ("mlp_cluster", "attn_heads"):
                continue
            if row.get("tier") != "screened":
                continue
            indices = torch.load(row["units"])["indices"]
            rows.append({"id": row["id"], "type": row["type"], "model": row["model"], "layer": int(row["layer"]),
                         "indices": [int(i) for i in torch.as_tensor(indices).reshape(-1).tolist()],
                         "mined_skill": (row.get("skill") or {}).get("name")})
    return rows


def components_for(pair, arm, data, seed, library):
    """The component dicts of one synth arm."""
    random_control = arm == "synth_random"
    comps = []
    if pair == "p1":
        host_layer = 21 if arm == "synth_top" else None
        comps.append({"id": "big_block", "kind": "block", "donor": BIG, "layers": [8, 20], "skill": "mmlu",
                      "texts": data["domain"]["big"], "host_layer": host_layer})
    elif pair == "p2":
        comps.append({"id": "math_block", "kind": "block", "donor": MATH_INSTRUCT, "layers": [8, 17],
                      "skill": "gsm8k", "texts": data["domain"]["math"]})
        comps.append({"id": "coder_block", "kind": "block", "donor": CODER_INSTRUCT, "layers": [14, 23],
                      "skill": "mbpp", "texts": data["domain"]["coder"]})
    else:
        for row in library:
            comps.append({"id": row["id"], "kind": row["type"], "donor": row["model"], "layer": row["layer"],
                          "indices": row["indices"], "skill": "gsm8k", "texts": data["domain"]["math"],
                          "host_layer": row["layer"]})
    for c in comps:
        c["control_seed"] = seed
        if random_control:
            c["control"] = "random"
    return comps


def baseline_components(pair, data):
    """Pseudo components for the LoRA arms: one donor and its texts per skill source."""
    if pair == "p1":
        return [{"id": "big", "donor": BIG, "skill": "mmlu", "texts": data["domain"]["big"]}]
    if pair == "p2":
        return [{"id": "math", "donor": MATH_INSTRUCT, "skill": "gsm8k", "texts": data["domain"]["math"]},
                {"id": "coder", "donor": CODER_INSTRUCT, "skill": "mbpp", "texts": data["domain"]["coder"]}]
    return [{"id": "math", "donor": MATH_BASE, "skill": "gsm8k", "texts": data["domain"]["math"]}]


# ------------------------------------------------------------------ cell


def lora_arm(model, tokenizer, pair, data, kd, seed, practice):
    guard = list(PAIRS[pair]["guard"])
    """LoRA on the base, with or without donor KL, same steps and guard as the synthesis heal."""
    from wsp import synth
    from wsp.assemble import load_donor

    comps = baseline_components(pair, data)
    budget = dict(HEAL, seed=seed, kd=kd, output_init="auto")
    s = synth.settings_of(budget, {})
    donor_map = {}
    for c in comps:
        if c["donor"] not in donor_map:
            donor_map[c["donor"]] = load_donor(c["donor"], "cuda")
    texts = {"align": data["align"], "general": data["general"]}
    enc, report = synth.prepare_data(tokenizer, comps, texts, s)
    base_scores = synth.score_practice(model, tokenizer, practice)
    steps = s["stage1_steps"] + s["stage2_steps"]
    heal = synth.lora_baseline(model, tokenizer, comps, donor_map, enc, s, steps)
    after = synth.score_practice(model, tokenizer, practice)
    bad = synth.guard_failures(after, base_scores, guard, s["tolerance"])
    out = {"heal": heal, "practice_base": base_scores, "practice_after": after, "guard_failed": bad,
           "rolled_back": False, "trainable": synth.param_counts(model)["lora"]}
    if bad:
        synth.strip_lora(model)
        out["rolled_back"] = True
    del donor_map
    return out


def run_cell(pair, arm, seed, library=None):
    import torch
    import transformers

    from wsp import skills_eval as ev
    from wsp import synth
    from wsp.practice import practice_sets, scoring_items

    start = time.time()
    torch.manual_seed(seed)
    spec = PAIRS[pair]
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec["base"])
    model = transformers.AutoModelForCausalLM.from_pretrained(spec["base"], dtype=torch.bfloat16).to("cuda").eval()
    with open(f"/vol/round20/data_{pair}.json") as f:
        data = json.load(f)
    skills = list(spec["skills"])
    scoring = scoring_items(ev, skills, heldout_mmlu=(pair == "p1"))
    practice_items = practice_sets(ev, skills)
    practice = {}
    for skill, items in practice_items.items():
        practice[skill] = {"items": items}
    row = {"pair": pair, "arm": arm, "seed": seed, "base": spec["base"], "targets": spec["targets"],
           "guard": spec["guard"], "heal_budget": HEAL}

    if arm == "base":
        pass
    elif arm in ("kd_lora", "lora"):
        row["lora_arm"] = lora_arm(model, tokenizer, pair, data, arm == "kd_lora", seed, practice)
        row["heal_seconds"] = row["lora_arm"]["heal"]["seconds"]
        row["trainable"] = row["lora_arm"]["trainable"]
    else:
        comps = components_for(pair, arm, data, seed, library)
        texts = {"align": data["align"], "general": data["general"]}
        budget = dict(HEAL, seed=seed)
        if arm == "synth_stage1":
            budget["stage2_steps"] = 0
        if arm == "synth_old_out":
            budget["output_init"] = "old"
        if arm == "synth":
            budget["control_run"] = True
        if pair == "p3":
            budget["components_per_step"] = 3
            budget["test_limit"] = 40
        model, tokenizer, report = synth.synthesize((model, tokenizer), comps, texts, budget, practice=practice,
                                                       guard=spec["guard"])
        row["report"] = report
        row["heal_seconds"] = report["heal_seconds"]

    counts = synth.param_counts(model)
    if arm not in ("base", "kd_lora", "lora"):
        row["trainable"] = counts["glue"] + counts["gates"] + row["report"]["planned_lora"]
    row["params_total"] = counts["total"]
    row["params_per_token"] = counts["per_token"]
    row["scores"] = {}
    for skill in skills:
        result = ev.evaluate_skill(model, tokenizer, skill, items=scoring[skill], batch_size=32)
        row["scores"][skill] = result["score"]
    row["seconds"] = round(time.time() - start, 1)
    return row


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=131072, timeout=6 * 3600)
def cell(pair, arm, seed, library=None):
    row = run_cell(pair, arm, seed, library)
    os.makedirs("/vol/results/round20", exist_ok=True)
    with open(f"/vol/results/round20/{pair}_{arm}_s{seed}.json", "w") as f:
        json.dump(row, f, default=str)
    volume.commit()
    return row


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, pairs_only: str = "", seeds_only: str = "", arms_only: str = ""):
    todo = []
    for c in cells():
        # Optional filters for a staged run: "p1,p2", "0", "synth,synth_random".
        if pairs_only and c[0] not in pairs_only.split(","):
            continue
        if seeds_only and str(c[2]) not in seeds_only.split(","):
            continue
        if arms_only and c[1] not in arms_only.split(","):
            continue
        todo.append(c)
    mine = []
    for index, c in enumerate(todo):
        if index % nshards == shard:
            mine.append(c)
    print(f"shard {shard}/{nshards}: {len(mine)} cells")
    pairs = sorted({c[0] for c in mine})
    library = library_rows() if "p3" in pairs else None
    for pair in pairs:
        prep.remote(pair)
    args = []
    for pair, arm, seed in mine:
        args.append((pair, arm, seed, library if pair == "p3" else None))
    rows = []
    for row in cell.starmap(args, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            continue
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ["pair", "arm", "seed", "scores", "params_total",
                                                  "params_per_token", "heal_seconds"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round20_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
