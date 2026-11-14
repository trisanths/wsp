# Agent brief: Arjun, RL environment and agent swarm (part 3)

You are the build agent for Arjun on a four-person hackathon team. Arjun already has agent-swarm infrastructure; build on it. Part 3 already works in the repo. Read this whole brief and issue trisanths/wsp#1 before you write code. Arjun owns the decisions; escalate to Arjun when this brief tells you to.

## The project

We are building a weight-space programming language. The goal is to find the best combinations of model parts, without regard to model family. It has three parts:

1. A library of reusable model parts, cut from open-weight models of any family and any size. Each part has a measured signature (what it does, where it lives).
2. A language and compiler. A person or agent writes a program that names a host model and a list of parts. `compile()` grafts the parts into the host and joins them with small learned glue layers. A short heal pass trains only the glue.
3. An RL-style environment where agents write programs to maximize benchmark scores at a fixed evaluation budget, compared against evolutionary search, LoRA, and full fine-tuning.

The output is one fixed combined model per task type. Routing at inference time is out of scope.

Part types:

- `mlp_cluster`: a set of MLP neurons from one donor layer. Built.
- `attn_heads`: a set of query heads (and the KV heads they use) from one donor layer. Being built now.
- `expert`: one MoE expert's MLP from one layer. Being built now.
- `block`: whole donor layers i to j. Later.

Deletion Tuning masks (Lu et al., arXiv:2611.10866) are now one way to carve a part out of a donor, not the goal. The mask algebra (`|`, `&`, `-`, `restrict`) and task vectors still work on the host. The 37 public Llama-3.2-1B masks stay in `catalog.jsonl`.

Judges reward measured gains on real benchmarks. Build the smallest version that produces real numbers.

## Status board

Part 3 already works in the repo. Its status board is GitHub issue trisanths/wsp#1. Post a short comment there when you hit a milestone or a gate, using the issue's template. Read the issue before you start so you do not repeat work.

## Team and ownership

| Person | Part | Owns |
|---|---|---|
| Person A | 1a. Parts catalog | Donor registry, part records, index files, repo verification |
| Person B | 1b. Eval and extraction | Eval harness, extraction of parts with known function, signatures, controls |
| Trisanth | 2. Language and compiler | Program API, `compile()`, grafts, heal; research direction |
| Arjun (you) | 3. RL environment and swarm | Env wrapper, planner and judge agents, baselines at equal budget |

## Shared interfaces

Do not change these without telling all four people.

1. Catalog record. One JSON object per line in `catalog.jsonl`. Person A writes the record, Person B fills `signature`, everyone reads it. A part record:
   ```json
   {"id": "qmath_l14_c3", "type": "mlp_cluster",
    "model": "Qwen/Qwen2.5-Math-1.5B", "layer": 14,
    "units": "clusters/qmath_l14_c3.pt",
    "trained_on": "GSM8K train activations",
    "signature": {"effect": {}, "fit_r2": {}}}
   ```
   The id and values above are an example, not a real record. `type` is `mlp_cluster`, `attn_heads`, `expert`, or `block`. Use `layer` for one layer and `layers` for a range. The older record types (`deletion_mask`, `task_vector`, `lora`) keep their fields.
2. Index file. A `.pt` dict with `kind`, `model`, `layer`, and `indices`. `indices` is a LongTensor of neuron ids (`mlp_cluster`), query-head ids (`attn_heads`), or expert ids (`expert`). For `attn_heads` the compiler derives the KV heads from the donor's GQA config.
3. Program (Trisanth owns). A Python object with a JSON form (`to_dict`, `from_dict`):
   - `Program(host=...)` (`base=` still works)
   - `.graft(id, at=None, control=None, control_seed=0)`
   - `.heal(steps, data, lr, batch_size, seq_len)`
   - masks and task vectors on the host: `use`, `|`, `&`, `-`, `restrict`, `.tv(id, weight)`
4. `compile(program) -> model_path` (Trisanth owns). Deterministic for a given program and seed. Saves `program.json` and `compile_report.json` next to the model.
5. `eval(model_path, split, tasks) -> scores` (Person B owns). Every call counts against one shared budget. Results are cached by (model hash, prompt). It must accept models made by `wsp.load`.

## Facts every agent needs

- The compiler fetches only the tensors a part needs from the donor's safetensors files, by HTTP range request. Never download a whole large model.
- A grafted model loads with `wsp.load(path)`, which returns `(model, tokenizer)`. It generates with Hugging Face `generate`, not vLLM. `attn_heads` grafts need `use_cache=False`.
- Splits for the first task (GSM8K). The search split is 200 items carved from GSM8K train (seeded, see `gsm8k_splits` in `wsp/data.py`). The heal data is the rest of GSM8K train. The held-out split is GSM8K test, run once at the end. The three never overlap.
- Required controls. Control 1 wraps the same glue around a random part of the same size from the same donor layer, with an identical heal (`Program.graft(..., control="random")`). Control 2 is a LoRA on the host, matched to glue plus part parameters. Every claim uses 3 seeds.
- Each `compile_report.json` has, per graft: `fit` (`ridge` or `skipped`), `fit_r2`, `fit_pairs`, `part_params`, `glue_params`, `host_layer`. It also has the heal losses. Log `fit_r2` next to every score.
- Compute is Modal, $250 per person, more on request. Everyone joins one shared Modal workspace, so the volume (models, compiled outputs) is shared.

## Donor pool

First pair: host `unsloth/Llama-3.2-1B-Instruct`, donor `Qwen/Qwen2.5-Math-1.5B`, task GSM8K.

Main pool is the 4B to 40B tier. These models fit on one GPU, so extraction is cheap.

- Dense: Qwen3.8 27B, Nanbeige4.3-33B, Gemma 4 31B, Muse Glimmer 30B, K2 Horizon 7B, Phi-4 14B, Mistral Small 3.1 24B.
- MoE (experts are ready-made parts): Qwen3.6 35B-A3B, K2 Horizon MoVA 36B-A4B, G9v3-39A5B, LFM2.5 24B-A2B.

Later, one expert from a frontier MoE (Kimi K3.1, DeepSeek V4.1, or GLM-5.3) to support the any-size claim.

These names come from a November 2026 leaderboard. Before you use any of them, verify the Hugging Face repo id, the license, and that the weights are published. Do not assume a name is correct.

## Your job

You make agents good at building models through the language, and you prove it with a budgeted comparison. Your code is already running on the real `compile()` and `eval()`. Read the status board at issue trisanths/wsp#1, post your milestones there with the issue's template, and use it to see what Trisanth and Person B have landed.

1. Action space. An action is a program: a host, a list of grafts, optional host masks and task vectors, and an optional heal pass. Grafts name catalog part ids, each with an optional host depth `at`. This is the real JSON that `Program.to_dict()` returns for one graft and a heal pass (the part id is a placeholder):
   ```json
   {"host": "unsloth/Llama-3.2-1B-Instruct", "seed": 0,
    "masks": null, "task_vectors": [],
    "grafts": [{"id": "qmath_l14_c3", "at": 9, "control": null, "control_seed": 0}],
    "heal": {"steps": 300, "data": "gsm8k_train", "lr": 0.0003,
             "batch_size": 8, "seq_len": 384}}
   ```
   Build it in Python with `Program(host=...).graft(id, at=9).heal(steps=300, data="gsm8k_train")`, and load a JSON program with `Program.from_dict`. `at=None` uses the default depth, `round(donor_layer / donor_layers * host_layers)`. A mask expression on the host still works through the `masks` field. `control="random"` swaps the part for random units of the same count from the same donor layer. The schema is locked in `wsp/program.py`. Read it, and do not invent fields.
2. Environment. `step(program) = eval(compile(program), "search")`. Observation: the catalog with signatures, the history of programs and scores, and the `fit_r2` per graft. Action: a program. Reward: combined normalized score. The episode ends when the budget is spent.
3. Cheap signal. Each compile reports `fit_r2` per graft before any heal, in `compile_report.json`. A low value means a poor depth pairing. Planners can use it to drop bad placements before they spend an eval. Log `fit_r2` next to every score so the swarm can learn how well it predicts the final score. When `fit` is `skipped` (the donor is too large to run), there is no signal, so treat those grafts as higher risk.
4. Cost per step. Compile plus heal takes minutes, and eval uses Hugging Face generation, not vLLM. Count compile and heal time against the budget as well as eval calls. Expect far fewer steps per hour than with mask-only programs. Size K and the step budget to match. `attn_heads` grafts run with `use_cache=False` and are slower still.
5. Agent roles on your swarm infra.
   - Analysts read signatures, `fit_r2`, and probe results (Person B's tools), and summarize the capability map.
   - Planners (parallel) read the catalog, the map, and the eval history, and propose K programs, each with a one-line hypothesis.
   - Judge ranks results on the search split, keeps a top-k frontier, kills dead branches, and enforces the budget.
   - Loop: analysts, then planners propose K, then drop low-`fit_r2` placements, then compile, then eval on search, then the judge updates the frontier. Repeat until the budget is spent. Then run the top 2 or 3 on the held-out split once.
   - Collapsed models are data. Log them so planners learn.
6. Baselines at the same budget. Each baseline's eval calls go through the same shared counter.
   - Evolutionary search over the same graft space (part ids, depths, heal settings). This is the primary baseline.
   - The random-part control: the same glue around a random part of the same size from the same donor layer, with an identical heal (`control="random"`). A swarm program that does not beat its random-part control has not shown that the part matters.
   - LoRA on the host and full fine-tuning on the same data, for the "versus regular training" comparison. The LoRA is matched to glue plus part parameters.
   - Claims use 3 seeds.
7. Demo outputs.
   - Chart 1: best combined score vs number of evaluations (and vs wall-clock cost), swarm curve vs evolutionary curve.
   - Chart 2: `fit_r2` vs final score across all compiles, to show whether the cheap signal works.
   - A scrolling agent log: hypothesis, program, `fit_r2`, score.
   - The pitch: one slide for the big vision (a catalog and language over parts from every open model, of any family and size), then the measured version.

## Claims

- Claim B (yours): at a fixed evaluation budget, the swarm matches or beats evolutionary search over the same graft space, or reaches a given score with N times fewer evals. Fallback headline: "same score, N times fewer evals."
- Claim A (Trisanth's): grafted parts from other families beat the random-part control at equal heal budget.
- Claim C (Trisanth's): the same pipeline works across families and sizes, down to one expert from a frontier MoE.

## Gates

- Hour 1: environment runs end to end on the real `compile()` and `eval()`, on the first pair (host Llama-3.2-1B-Instruct, donor Qwen2.5-Math-1.5B, GSM8K).
- Hour 8: evolutionary baseline running in the background on the real stack, and the random-part control wired in.
- Hour 16: first swarm vs baseline curve at a fixed budget, with 3 seeds if the budget allows.
- End: curves, agent log, and pitch.

## Rules

- Budget accounting. Every eval call, swarm or baseline, increments the one shared counter. Compile and heal time count too. The comparison is only valid at equal budgets.
- Never report search-split numbers as results. Final numbers come from the held-out split (GSM8K test), run once.
- Write plain, readable Python: simple loops and ifs. Avoid dense comprehensions, `setdefault`, and heavy type annotations. Arjun reads and steers the code.
- Fixed seeds. Append each step to the shared `experiments.jsonl` (timestamp, agent, hypothesis, program, split, scores, `fit_r2`, eval count).
- Report failures. Never invent a number.
- Never download a whole large model. Compiles fetch only part tensors, on Modal.
- Verify donor repo ids and licenses before you let planners use a donor the catalog does not list. Only use parts that are in `catalog.jsonl`.
- Escalate to Arjun when a gate slips by more than one hour, a result contradicts a claim, or the budget is too small to separate the curves. Modal is $250 per person on one shared workspace.
- Do not run `pkill` or `killall` by name. Do not touch other people's processes or files.

## Open questions (Arjun will settle these)

- Which benchmarks the hackathon tracks score.
- GPU type and hours available beyond the $250 per person.
- What in Arjun's original team doc should merge into this plan.
