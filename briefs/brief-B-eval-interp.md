# Agent brief: Person B, eval and extraction (part 1b)

You are the build agent for Person B on a four-person hackathon team. Read this whole brief before you write code. Your human owns the decisions; escalate to them when this brief tells you to.

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
| Person B (you) | 1b. Eval and extraction | Eval harness, extraction of parts with known function, signatures, controls |
| Trisanth | 2. Language and compiler | Program API, `compile()`, grafts, heal; research direction |
| Arjun | 3. RL environment and swarm | Env wrapper, planner and judge agents, baselines at equal budget |

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

You own the measuring stick and the search for parts with known function. If the eval harness is late or wrong, nothing else in the project counts. Build it first.

1. Eval harness (hours 0 to 8, top priority). Implement interface 5.
   - It accepts a model path made by `wsp.load` and a plain Hugging Face model. Grafted models generate with Hugging Face `generate`. `attn_heads` grafts need `use_cache=False`. This is slower than vLLM, so keep the search split small. You may keep vLLM for plain host or masked models if it is faster, but the same prompts and templates apply to every model.
   - Greedy decoding, fixed seeds, the same prompts and templates for every model.
   - Splits for GSM8K: search is 200 items from `gsm8k_splits` in `wsp/data.py`. Heal data is the rest of GSM8K train and is not yours to score on. Held-out is GSM8K test, run once at the end. Never overlap them. For other tasks the human picks, build the same three-way split.
   - Report scores for host, control, and graft side by side. Combined score is the mean of per-skill scores, each normalized to [0, 1] between the host and the best reference model. Also report each skill alone.
   - A budget counter shared by the swarm and every baseline. Every call increments it. The comparison is only valid at equal budgets.
   - A cache keyed by (model hash, prompt).
   - Log `fit_r2` from `compile_report.json` next to every score.
   - First numbers: the host, the donor, the no-graft heal control, and one compiled graft on the search split.
2. Extraction. Find parts with known function in donor models. The parts are MLP neuron clusters, query-head sets, and MoE experts. Write each one as an index file (`.pt` dict with `kind`, `model`, `layer`, `indices`) and hand it to Person A for the catalog.
   - Start from `wsp/extract.py`. It has stub extractors: `top_activation_neurons` (the top n neurons by mean absolute activation on task text) and `top_routed_experts` (the experts with the highest mean routing probability on task text). Read the code before you use them. Replace or extend them where they fall short.
   - Run extraction on task text, such as the GSM8K train activations, for the first pair. Then run it on the main-pool donors. The 4B to 40B tier fits one GPU.
   - For MoE donors, measure the mean routing probability per expert on the task, and which experts hurt most when ablated.
   - Compare against a random part of the same size from the same layer. A part with no gain over random has no known function. Do not write it as a useful part.
3. Probe tool. `probe(part, layers) -> effect dict`: remove the part from a compiled model (or ablate the units in the donor) and measure the drop on each benchmark.
4. Controls. Build the control programs the team needs: `Program.graft(..., control="random")` with an identical heal, and a LoRA on the host matched to glue plus part parameters. Every claim uses 3 seeds. Report the spread across seeds.
5. Interp agent loop. An agent that chooses probe and extraction experiments, runs them, and writes results into each catalog record's `signature`. It builds a capability map ("skill X lives in donor layers 12 to 16"). Log each experiment as hypothesis, action, result.
6. Overlap and masks. The overlap tool `overlap(a, b, by='layer')` (Jaccard and counts of shared deletions per layer and module) still works for same-base masks. Keep it for the 37 public masks.
7. Demo visuals. Fit R squared by depth pairing, score by graft against control, and per-layer part locations in each donor.

## Gates

- Hour 1: `eval()` stub committed that returns fake scores in the real format.
- Hour 8: real `eval()` scores the host, the donor, the no-graft control, and one compiled graft on the search split. This is the most important gate in the project.
- Hour 16: index files for at least 3 parts per donor on 3 donors, signatures filled, and the random-part control run on each.
- End: held-out evaluation, run once, for the final table, with 3 seeds.

## Rules

- Never report search-split numbers as results. Label any search-split number as such.
- Never tune on the held-out split. Never train the glue on the search split.
- Write plain, readable Python: simple loops and ifs. Avoid dense comprehensions, `setdefault`, and heavy type annotations. Your human reads and steers the code.
- Fixed seeds. Append each run to the shared `experiments.jsonl` (timestamp, model, split, scores, `fit_r2`, eval count).
- Never download a whole large model. The compiler fetches part tensors by range request. Extraction runs the full donor on Modal, so check size against budget first.
- Verify each donor repo id, license, and published weights before use. The names in the donor pool are from a November 2026 leaderboard and are not checked.
- Report failures. Never invent a number.
- Post milestones on issue trisanths/wsp#1.
- Escalate to your human when a gate slips by more than one hour, a result contradicts this brief, or eval time or Modal budget makes the plan unworkable.
- Do not run `pkill` or `killall` by name. Do not touch other people's processes or files.

## Open questions (your human will settle these)

- Which benchmarks the hackathon tracks score.
- GPU type and hours available beyond the $250 per person.
