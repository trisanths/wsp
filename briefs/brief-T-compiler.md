# Agent brief: Trisanth, language and compiler (part 2)

You are the build agent for Trisanth on a four-person hackathon team. Trisanth also sets research direction for the team. Read this whole brief, `README.md`, and `docs/superpowers/specs/2026-11-14-mixed-family-compiler-design.md` before you write code. Trisanth owns the decisions; escalate to Trisanth when this brief tells you to.

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
| Trisanth (you) | 2. Language and compiler | Program API, `compile()`, grafts, heal; research direction |
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

## What is built

- Mask algebra and task vectors on the host: `|`, `&`, `-`, `restrict`, `.tv(id, weight)`.
- `Program(host=...)` (`base=` is an alias), `.graft(id, at=None, control=None, control_seed=0)`, `.heal(steps, data, lr, batch_size, seq_len)`, and the JSON form (`to_dict`, `from_dict`). Old `program.json` files still load.
- `compile()` order: task vectors, then masks, then grafts, then heal. It saves a Hugging Face model, `program.json`, and `compile_report.json`.
- The `mlp_cluster` part, built from a donor's own layer code with only the needed tensors fetched by range request. Glue is `W_in` (host size to donor size), the part, and `W_out` (donor size to host size) on a skip connection, with `W_out` starting at zero.
- The ridge fit for `W_in` on offset-matched hidden states, with fit R squared in the report.
- The random-part control (`control="random"`) and a heal pass that trains only the glue.
- `wsp.load(path)` returning `(model, tokenizer)`, and GSM8K splits in `wsp/data.py`.
- Stub extractors in `wsp/extract.py`: `top_activation_neurons` and `top_routed_experts`.
- Parts code for `expert` and `attn_heads` exists in `wsp/parts.py`. Check what is wired into `compile()` before you rely on it.

Each graft in `compile_report.json` has `fit` (`ridge` or `skipped`), `fit_r2`, `fit_pairs`, `part_params`, `glue_params`, and `host_layer`. The report also has the heal losses.

## What is next

1. Finish `attn_heads` and `expert` parts end to end: build, ridge fit, save, reload through `wsp.load`. `attn_heads` need `use_cache=False` in generation. Index files hold query-head ids, and you derive the KV heads from the donor's GQA config.
2. `block` parts (whole donor layers i to j). Default placement uses layer i. This is the coarse fallback when fine parts do not hold their function outside their model.
3. Real-model Modal runs. First run: host `unsloth/Llama-3.2-1B-Instruct`, donor `Qwen/Qwen2.5-Math-1.5B`, GSM8K. Compare host, host plus no-graft heal, the random-part control, and the real graft, on the search split, with Person B's `eval()`. This is the hour-8 gate.
4. Donor pool. Extend to the main pool (4B to 40B dense and MoE donors, listed above) and then one frontier MoE expert. When a donor is too large to run, `W_in` starts random and the report marks the fit as `skipped`. Verify each donor's repo id, license, and published weights before use.
5. Second control. Add the LoRA on the host matched to glue plus part parameters, and make the report print both controls next to the graft.
6. Cross-family claim. Run the cross-family and any-size tests described under Claims.
7. Text syntax (only if time). A small parser from plain text into the Program object. The Python API is the real interface.

## Claims

- Claim A: grafted parts from other families beat the random-part control at equal heal budget. The control wraps the same glue around a random part of the same size from the same donor layer, with identical heal steps and data. The second control is the matched LoRA. A negative result is reportable: "the gain comes from the glue and heal, not the part," with the fit R squared and part locations as evidence. Claims use 3 seeds.
- Claim B (Arjun's): an agent swarm reaches a given combined score with fewer evaluations than evolutionary search over the same graft space.
- Claim C: the pipeline works across families and sizes. A part from a 4B to 40B donor helps a small host, and one expert cut from a frontier MoE joins the same pipeline, because the compiler fetches only that expert's tensors.
- Related work, to cite and not claim as new: model stitching, layer-aware merging, and "analyze activations, then transplant" (ARM, ACM, GPTailor, Sakana evolutionary merge). Deletion Tuning masks (Lu et al., arXiv:2611.10866) stay as one way to carve a part.

## Splits and honesty rules

- Search: 200 items carved from GSM8K train (seeded, `gsm8k_splits`). Heal data: the rest of GSM8K train. Held-out: GSM8K test, run once at the end. They never overlap.
- A gain from heal alone is not a result. Every graft result goes beside the control with the same heal.
- Never report search-split numbers as results.

## Gates

- Hour 1: `Program` and `compile()` interfaces committed. They are.
- Hour 8: the first pair compiled on Modal, healed, and scored by Person B's real `eval()` for host, control, and graft. The pipeline running end to end is the gate. Beating the control is the claim, not the gate.
- Hour 16: `attn_heads` and `expert` parts working, at least one main-pool donor, and the first Claim A table with 3 seeds.
- End: heal the top 2 or 3 programs; held-out numbers via Person B.

## Rules

- Write plain, readable Python: simple loops and ifs. Avoid dense comprehensions, `setdefault`, and heavy type annotations. Trisanth reads and steers the code.
- Fixed seeds. Every compiled model is saved with the program that produced it. Append each run to the shared `experiments.jsonl` (timestamp, program, hypothesis, split, scores, `fit_r2`, eval count).
- Never download a whole large model. Fetch the header and the needed tensors by range request.
- Report failures. A collapsed model is data: log it.
- Post milestones on issue trisanths/wsp#1.
- Escalate when a gate slips by more than one hour, a result contradicts a claim, or compute is too small. Modal is $250 per person on one shared workspace.
- Do not run `pkill` or `killall` by name. Do not touch other people's processes or files.
