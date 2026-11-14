# Agent brief: Trisanth, language and compiler (part 2)

You are the build agent for Trisanth on a four-person hackathon team. Trisanth also sets research direction for the team. Read this whole brief before you write code. Trisanth owns the decisions; escalate to Trisanth when this brief tells you to.

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
| Person B | 1b. Eval and interp | Eval harness, probe and overlap tools, interp agent loop, signatures |
| Trisanth (you) | 2. Language and compiler | Program API, `compile()`, stitching, heal; research direction |
| Arjun | 3. RL environment and swarm | Env wrapper, planner and judge agents, baselines at equal budget |

## Shared interfaces (frozen in hour 1)

Do not change these without telling all four people. Commit a working stub for the ones you own in hour 1.

1. **Catalog record** (Person A writes, Person B fills `signature`, everyone reads). One JSON object per line in `catalog.jsonl`:
   ```json
   {"id": "mask_mmlupro_biology_llama1b", "type": "deletion_mask",
    "base": "meta-llama/Llama-3.2-1B-Instruct", "family": "llama3.2",
    "trained_on": "MMLU-Pro biology SFT", "path": "components/...",
    "layers": [0, 1, 2], "num_deleted": 41237,
    "signature": {"effect": {}, "deleted_per_layer": [], "overlap_with": {}}}
   ```
   `type` is one of `deletion_mask`, `task_vector`, `lora`, `stitch`.
2. **Program** (you own). A Python object built from catalog ids: `use`, `|` (union of deletions), `&` (intersection), `-` (difference), `restrict(layers)`, `tv(id, weight)`, `stitch(id, at_layer)`, `heal(steps, data)`.
3. **`compile(program) -> model_path`** (you own). Deterministic for a given program and seed. Saves the program next to the model.
4. **`eval(model_path, split, tasks) -> scores`** (Person B owns). Every call counts against one shared budget. Results are cached by (model hash, prompt).

## Your job

You turn programs into working models. Arjun's environment and Person B's evals both call your `compile()`.

1. **Program API (hours 1 to 8).** Interface 2 as a plain Python class over boolean tensors. No parser yet; agents call the API directly.
   - `|`: delete a weight if any operand deletes it.
   - `&`: delete only weights that all operands delete.
   - `-`: deletions in A that are not in B.
   - `restrict(layers)` and a `drop ... where overlap(...)` rule: filter deletions by layer, module, or overlap.
   - `tv(id, weight)`: add a layer- or module-sliced task vector through mergekit, applied before masks.
2. **Same-base compile (hours 1 to 8).** Apply the combined mask to the base (reuse `src/apply_mask.py` from https://github.com/agentic-learning-ai-lab/deleting-weights, MIT license) plus any mergekit task vectors. Save a standard Hugging Face model and the program beside it.
3. **Heal pass.** After assembly, optionally fine-tune the combined mask (learning bits only, so it is cheap) or a small LoRA on mixed data. Report every result with and without heal. Deletion Tuning sparsity is emergent (no fixed k), so a `budget deleted <= X%` clause needs a rule: drop the lowest-logit deletions, or add a penalty. Pick one and document it.
4. **Cross-family stitching (hours 8 to 16).** `stitch(id, at_layer)`: a learned linear map between two models' residual streams at a layer, trained during compile. This is the stretch claim. Do not let it block same-base work.
5. **Text syntax (only if time).** A small parser from the plain-text syntax into the Program object:
   ```
   base meta-llama/Llama-3.2-1B-Instruct
   use bio = mask_mmlupro_biology_llama1b
   use chem = mask_mmlupro_chemistry_llama1b
   program = (bio | chem)
   restrict program layers 4..24
   compile
     heal mask_finetune steps 300
   ```
6. **Research direction.** Run the composition experiments for Claim A: union, intersection, difference, and layer-restricted union over pairs of masks, with and without heal, scored on each parent's task and on non-target tasks for forgetting. Compare against TIES over full fine-tunes and merged LoRAs.

## Claims

- **Claim A:** independently trained deletion masks on one base compose with simple mask algebra, with less forgetting than merging fine-tuned checkpoints or LoRAs. A negative result is still reportable: "masks do not compose, and the overlap structure explains why."
- **Claim B (Arjun's):** an agent swarm reaches a given combined score with fewer evaluations than evolutionary merging.
- **Stretch Claim C:** a component from a different family, joined by a stitching layer, measurably helps on a benchmark.
- Not novel alone: layer-aware merging and "analyze activations, then transplant" (ARM, ACM, GPTailor, Sakana evolutionary merge). Cite them.

## Gates

- Hour 1: Program class and a `compile()` stub committed. The stub returns the base model path.
- Hour 8: real `compile()` on same-base masks, scored by Person B's real `eval()`. This is the most important gate.
- Hour 16: first composition table; stitching attempted.
- End: heal the top 2 or 3 programs; held-out numbers via Person B.

## Rules

- Write plain, readable Python: simple loops and ifs. Avoid dense comprehensions, `setdefault`, and heavy type annotations. Trisanth reads and steers the code.
- Fixed seeds. Every compiled model is saved with the program that produced it. Append each run to the shared `experiments.jsonl` (timestamp, program, hypothesis, split, scores, eval count).
- Report failures. A collapsed merge is data, not a failure: log it.
- Escalate when a gate slips by more than one hour, a result contradicts a claim, or compute is too small.
- Do not run `pkill` or `killall` by name. Do not touch other people's processes or files.
