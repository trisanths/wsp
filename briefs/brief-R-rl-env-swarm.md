# Agent brief: Arjun, RL environment and agent swarm (part 3)

You are the build agent for Arjun on a four-person hackathon team. Arjun already has agent-swarm infrastructure; build on it. Read this whole brief before you write code. Arjun owns the decisions; escalate to Arjun when this brief tells you to.

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
| Trisanth | 2. Language and compiler | Program API, `compile()`, stitching, heal; research direction |
| Arjun (you) | 3. RL environment and swarm | Env wrapper, planner and judge agents, baselines at equal budget |

## Shared interfaces (frozen in hour 1)

Do not change these without telling all four people.

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

You make agents good at building models through the language, and you prove it with a budgeted comparison. Start on stubs in hour 1; switch to the real `compile()` and `eval()` when they land (target hour 8).

1. **Environment.** `step(program) = eval(compile(program), "search")`. Observation: the catalog with signatures, plus the history of programs and scores. Action: a program. Reward: combined normalized score. The episode ends when the eval budget is spent.
2. **Agent roles on your swarm infra.**
   - Analysts: read signatures and probe results (Person B's tools), and summarize the capability map.
   - Planners (parallel): read the catalog, the map, and the eval history; propose K programs, each with a one-line hypothesis.
   - Judge: ranks results on the search split, keeps a top-k frontier, kills dead branches, enforces the budget.
   - Loop: analysts, then planners propose K, then compile, then eval on search, then the judge updates the frontier. Repeat until the budget is spent. Then heal the top 2 or 3 and evaluate held-out once.
   - Collapsed merges are data. Log them so planners learn ("deleting overlap in late layers kills instruction following").
3. **Baselines at the same budget.** Evolutionary merge search (mergekit-evolve or a Sakana-style search over the same components) is the primary baseline. Also LoRA and full fine-tuning on the same data, for the "versus regular training" comparison. Every baseline eval call goes through the same shared counter.
4. **Demo outputs.**
   - Chart 1: best combined score vs number of evaluations, swarm curve vs evolutionary curve.
   - A scrolling agent log: hypothesis, program, score.
   - The pitch: one slide for the big vision (a catalog and language over all open models), then the measured version.

## Claims

- **Claim B (yours):** at a fixed evaluation budget, the swarm matches or beats evolutionary or zero-order merging, or reaches a given score with N times fewer evals. Fallback headline: "same score, 5x fewer evals."
- Claim A (Trisanth's): skill masks compose with less forgetting than merging.

## Gates

- Hour 1: environment runs end to end on stub `compile()` and `eval()`.
- Hour 8: evolutionary baseline running in the background on the real stack.
- Hour 16: first swarm vs baseline curve at a fixed budget.
- End: curves, agent log, and pitch.

## Rules

- Budget accounting: every eval call, swarm or baseline, increments the one shared counter. The comparison is only valid at equal budgets.
- Never report search-split numbers as results. Final numbers come from the held-out split, run once.
- Write plain, readable Python: simple loops and ifs. Avoid dense comprehensions, `setdefault`, and heavy type annotations. Arjun reads and steers the code.
- Fixed seeds. Append each step to the shared `experiments.jsonl` (timestamp, agent, hypothesis, program, split, scores, eval count).
- Report failures. Never invent a number.
- Escalate to Arjun when a gate slips by more than one hour, a result contradicts a claim, or the eval budget is too small to separate the curves.
- Do not run `pkill` or `killall` by name. Do not touch other people's processes or files.

## Open questions (Arjun will settle these)

- Which benchmarks the hackathon tracks score.
- GPU type and hours available.
- What in Arjun's original team doc should merge into this plan.
