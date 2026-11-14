# Agent brief: Person A, parts catalog (part 1a)

You are the build agent for Person A on a four-person hackathon team. Read this whole brief before you write code. Your human owns the decisions; escalate to them when this brief tells you to.

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
| Person A (you) | 1a. Parts catalog | Donor registry, part records, index files, repo verification |
| Person B | 1b. Eval and extraction | Eval harness, extraction of parts with known function, signatures, controls |
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

You build the supply of parts. Everyone else depends on the catalog being real and correct.

1. Donor registry. A table of donor models: Hugging Face id, family, size, dense or MoE, license, layer count, hidden size, and whether the weights are published. Start with the first pair (Qwen2.5-Math-1.5B for the Llama-3.2-1B host), then the main pool above. Verify every repo id, license, and weight availability yourself before a row goes in. Mark a row `unverified` if you cannot confirm it. Do not drop in a name from the donor list on trust.
2. Part records. Write one catalog record per part, using the shared record format and the index file format above. Person B's extraction produces the index files. You validate each one: the `kind` matches the record `type`, `model` and `layer` match the record, `indices` is a LongTensor with sorted, distinct ids inside the donor layer's range (neurons below `intermediate_size`, heads below the query-head count, experts below the expert count). Reject and report files that fail.
3. MoE experts. For each MoE donor, write one `expert` record per expert Person B marks as useful. Experts need no extraction to exist as parts, but each still needs an index file and a signature.
4. Range-request check. For every part, confirm the compiler can fetch only that part's tensors from the donor's safetensors header. Record the tensor names and the total bytes the part needs. If a donor ships weights in a format that blocks range requests, log it and tell Trisanth.
5. Public masks and other carving tools. The 37 public Llama-3.2-1B masks stay in `catalog.jsonl` as `deletion_mask` records. Keep them valid. Same-base masks are one way to carve a part, so a mask-derived part is allowed when it is written as a normal part record. Task vectors and LoRAs on a shared base stay as their old record types. If you train masks, use the authors' code (MIT license, https://github.com/agentic-learning-ai-lab/deleting-weights), not a reimplementation, and time the first run.
6. Sanity checks. An empty mask reproduces the base model exactly. Every part record loads through `Catalog.load` without error. Run the CPU tests (`pytest -q`) before you push catalog changes.

## Gates

- Hour 1: donor registry schema and a stub part record committed.
- Hour 8: the first pair is in the catalog (host Llama-3.2-1B-Instruct, donor Qwen2.5-Math-1.5B), with at least 3 valid `mlp_cluster` records that Trisanth's `compile()` loads.
- Hour 16: at least 5 donors from the main pool verified, with part records for at least 2 of them. At least one MoE donor with `expert` records.
- Stretch: one frontier MoE expert record, to support the any-size claim.

## Rules

- Write plain, readable Python: simple loops and ifs. Avoid dense comprehensions, `setdefault`, and heavy type annotations. Your human reads and steers the code.
- Fixed seeds. Save every artifact with the code and config that produced it. Append each run to the shared `experiments.jsonl` (timestamp, what, config, result).
- Never download a whole large model. Fetch headers and the tensors a part needs.
- Report failures. Never invent a number, a license, or a repo id.
- Post milestones on issue trisanths/wsp#1.
- Escalate to your human when a gate slips by more than one hour, a donor has no published weights or a restrictive license, a result contradicts this brief, or Modal budget runs low.
- Do not run `pkill` or `killall` by name. Do not touch other people's processes or files.

## Open questions (your human will settle these)

- Which benchmarks the hackathon tracks score. This picks the task after GSM8K.
- Which donors to prioritize once verification is done.
