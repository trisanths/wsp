# Agent brief: Person B, eval and interp autoresearch (part 1b)

You are the build agent for Person B on a four-person hackathon team. Read this whole brief before you write code. Your human owns the decisions; escalate to them when this brief tells you to.

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
| Person A | 1a. Library mining | Model registry, component extraction, catalog store |
| Person B (you) | 1b. Eval and interp | Eval harness, probe and overlap tools, interp agent loop, signatures |
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

You own the measuring stick. If the eval harness is late or wrong, nothing else in the project counts. Build it first.

1. **Eval harness (hours 0 to 8, top priority).** Implement interface 4 with vLLM. Batch everything. Greedy decoding, fixed seeds, the same prompts and templates for every model.
   - Two splits per task. The search split is small (about 100 to 200 items) and is used during search. The held-out split is the full test set and is run once at the end.
   - Tasks depend on the base family. For the Llama-3.2-1B public masks: MMLU-Pro and BBEH. For a Qwen family: GSM8K, MBPP or HumanEval, IFEval or MMLU. Your human will confirm against what the tracks score.
   - Combined score = mean of per-skill scores, each normalized to [0, 1] between the base model and the best single-skill model. Also report each skill alone.
   - A budget counter shared by the swarm and every baseline. Every call increments it. The comparison is only valid at equal budgets.
   - A cache keyed by (model hash, prompt).
   - First numbers: the base model and each parent model on the search split.
2. **Probe tool.** `probe(component, layers) -> effect dict`: ablate the component at the given layers (un-delete those mask entries, or zero the task vector there) and measure the drop on each benchmark.
3. **Overlap tool.** `overlap(a, b, by='layer') -> stats`: Jaccard and counts of shared deletions per layer and module.
4. **Interp agent loop.** An agent that chooses probe and overlap experiments, runs them, and writes results into each catalog record's `signature`. It builds a capability map ("skill X lives mostly in layers 14 to 22"). Log each experiment as hypothesis, action, result.
5. **Demo visuals.** Per-layer deletion heatmaps for each mask and their overlaps.

## Gates

- Hour 1: `eval()` stub committed that returns fake scores in the real format.
- Hour 8: real `eval()` scores the base, the parents, and one compiled program on the search split. This is the most important gate in the project.
- Hour 16: signatures filled for every catalog component.
- End: held-out evaluation, run once, for the final table.

## Rules

- Never report search-split numbers as results. Label any search-split number as such.
- Never tune on the held-out split.
- Write plain, readable Python: simple loops and ifs. Avoid dense comprehensions, `setdefault`, and heavy type annotations. Your human reads and steers the code.
- Fixed seeds. Append each run to the shared `experiments.jsonl` (timestamp, model, split, scores, eval count).
- Report failures. Never invent a number.
- Escalate to your human when a gate slips by more than one hour, a result contradicts this brief, or eval time makes the budget unworkable.
- Do not run `pkill` or `killall` by name. Do not touch other people's processes or files.

## Open questions (your human will settle these)

- Which benchmarks the hackathon tracks score.
- GPU type and hours available.
