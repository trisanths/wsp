# Agent brief: Person A, library mining (part 1a)

You are the build agent for Person A on a four-person hackathon team. Read this whole brief before you write code. Your human owns the decisions; escalate to them when this brief tells you to.

## The project

We are building a weight-space programming language. It has three parts:

1. A library of reusable model components, mined from many open-weight models, each with a measured signature (what it does, where it lives).
2. A language and compiler. A person or agent writes a program that references components; `compile()` assembles one working model. Within one base model it uses mask algebra and task-vector merging. Across model families it trains small stitching layers. A short heal pass fuses the result.
3. An RL-style environment where agents write programs to maximize benchmark scores at a fixed evaluation budget, compared against evolutionary merging, LoRA, and full fine-tuning.

The primary component is a Deletion Tuning mask: a binary mask over the gate_proj weights that zeroes fewer than 0.05% of all model parameters and matches full fine-tuning (Lu et al., arXiv:2611.10866). The open question we attack is whether masks for different skills compose. The paper does not test this; it names mask union only as future work.

Judges reward measured gains on real benchmarks. Build the smallest version that produces real numbers.

## Team and ownership

| Person | Part | Owns |
|---|---|---|
| Person A (you) | 1a. Library mining | Model registry, component extraction, catalog store |
| Person B | 1b. Eval and interp | Eval harness, probe and overlap tools, interp agent loop, signatures |
| Trisanth | 2. Language and compiler | Program API, `compile()`, stitching, heal; research direction |
| Arjun | 3. RL environment and swarm | Env wrapper, planner and judge agents, baselines at equal budget |

## Shared interfaces (frozen in hour 1)

Do not change these without telling all four people. Commit a working stub for the one you own in hour 1.

1. **Catalog record** (Person A writes, Person B fills `signature`, everyone reads). One JSON object per line in `catalog.jsonl`:
   ```json
   {"id": "mask_mmlupro_biology_llama1b", "type": "deletion_mask",
    "base": "meta-llama/Llama-3.2-1B-Instruct", "family": "llama3.2",
    "trained_on": "MMLU-Pro biology SFT", "path": "components/...",
    "layers": [0, 1, 2], "num_deleted": 41237,
    "signature": {"effect": {}, "deleted_per_layer": [], "overlap_with": {}}}
   ```
   `type` is one of `deletion_mask`, `task_vector`, `lora`, `stitch`.
2. **Program** (Trisanth owns). A Python object built from catalog ids: `use`, `|` (union of deletions), `&` (intersection), `-` (difference), `restrict(layers)`, `tv(id, weight)`, `stitch(id, at_layer)`, `heal(steps, data)`.
3. **`compile(program) -> model_path`** (Trisanth owns). Deterministic for a given program and seed. Saves the program next to the model.
4. **`eval(model_path, split, tasks) -> scores`** (Person B owns). Every call counts against one shared budget. Results are cached by (model hash, prompt).

## Your job

You build the supply of components. Everyone else depends on the catalog being real and correct.

1. **Model registry.** A table of models grouped by base family. Start with two families and about 10 models. Grow toward 100 only after the hour-8 gate. Same-base fine-tunes are the cheap case (they line up with no stitching), so most of the pool should be fine-tunes of a few bases.
2. **Public masks first.** Download the released masks from Hugging Face `Jacklu0831/deletion-tuning-masks`. The `sft/llama-3.2-1b-instruct-mmlu-pro` (14 subjects) and `sft/llama-3.2-1b-instruct-bbeh` (23 tasks) sets give 37 masks on one base. The 3B sets give 37 more. Use `src/apply_mask.py` from https://github.com/agentic-learning-ai-lab/deleting-weights to apply a mask. Write one catalog record per mask.
3. **Our own masks.** Train masks with the authors' code (MIT license), not a reimplementation. Their recipe: one logit per gate_proj weight, soft mask `W * sigmoid(alpha(t) * G)` in training, hard mask `W * 1[G > 0]` at eval, Wanda-score init, alpha ramp 1e2 to 1e3 for SFT, lr 5e-4, 400 steps. Time the first run and report the wall-clock; it decides how many skills we can train.
4. **Task vectors and LoRAs.** For public fine-tunes on the same base, store (fine-tuned minus base), sliced by layer range and module (attention or MLP). Note: Qwen2.5-Math-1.5B changed rope_theta (1e4 vs 1e6) and was heavily continued-pretrained, so its delta is not a clean fine-tune.
5. **Sanity checks.** An empty mask reproduces the base model exactly. Record deletions per layer for every mask.

## Gates

- Hour 1: catalog schema and a stub `catalog.jsonl` committed.
- Hour 8: at least 10 real components in the catalog, loadable by Trisanth's `compile()`.
- Hour 16: own masks trained for the target benchmarks; task vectors for the second family.

## Rules

- Write plain, readable Python: simple loops and ifs. Avoid dense comprehensions, `setdefault`, and heavy type annotations. Your human reads and steers the code.
- Fixed seeds. Save every artifact with the code and config that produced it. Append each run to the shared `experiments.jsonl` (timestamp, what, config, result).
- Report failures. Never invent a number.
- Escalate to your human when a gate slips by more than one hour, a result contradicts this brief, or compute is too small to train masks for the target skills.
- Do not run `pkill` or `killall` by name. Do not touch other people's processes or files.

## Open questions (your human will settle these)

- Which benchmarks the hackathon tracks score. This picks the first two families.
- GPU type and hours available.
