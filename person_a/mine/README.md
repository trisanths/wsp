# mine/: v2 parts-mining pipeline

Teammates: follow `RUNBOOK.md` at the repo root. This file is a developer reference.

## What it does
For one donor model, a weight-space programming agent (`gpt-6-astra`):
1. reads the raw weights in a network-blocked, read-only Modal sandbox;
2. hypothesises open-vocabulary labels;
3. writes frozen tests for those labels;
4. screens candidate parts by zero-ablation on a Modal GPU.

Every ablation scores all active skills in one teacher-forced pass, measured as the mean log-prob of the answer tokens.

## Files
| File | Role |
|---|---|
| `engine_app.py` | Deployed Modal class `Engine(model_key, hf_id, revision)`. Loads the donor and scores skills. Runs ablations, random baselines, activation/attribution/routing scans, and the optional math generation check. The client chooses the GPU per call through `with_options`. |
| `adapter.py` | Discovers layers, attention out-projections (heads), MLP down-projections and MoE expert modules, then installs zero-ablation hooks (see below). |
| `client.py` | `Miner`: pinned revision, state, skill payloads, light gate, catalog and `.pt` writer. |
| `evidence.py` | Evidence fields of a catalog row, from stored numbers: z-score against random parts, guard, `selection_data`, expert routing. |
| `enrich_rows.py` | Backfills those fields into existing `catalog/<model>.jsonl` rows. No GPU call unless `--routing` is given. |
| `skills.py` | `define_skill`: agent-written tests, with the generator run in the sandbox, a matched control, a chance check and freezing. |
| `agent.py` | OpenAI Responses tool loop (weight-first unlock, budgets, notes, cost log). |
| `weight_sandbox.py`, `wsp_helpers.py` | The weight sandbox. Its weights come from the engine's HF cache on the `swarmy-models` volume, at the pinned snapshot. |
| `build_data.py` | Builds `data/builtin_skills.json` (math, code, facts, multilingual, general_text; 100 search + 50 held-out each). |
| `run.py` / `run_many.py` | Entry points. `--smoke` checks the setup without the agent. |
| `compile_catalog.py` | Merges `catalog/*.jsonl` into `catalog.jsonl`. |

## How each part kind is switched off (zero ablation)
- **`mlp_cluster`:** zeroes those columns of the input to the MLP down-projection. For MoE layers with a shared expert, this means the shared expert's down-projection.
- **`attn_heads`:** zeroes each head's slice of the input to the attention out-projection. Head size is read from the q/k norm when present. This also covers linear-attention (Gated DeltaNet) layers through their `out_proj`.
- **`expert`:** sets the routing weight of the banned experts to 0 at the input of the fused experts module (transformers-5 `experts(hidden, top_k_index, top_k_weights)`). The other selected experts keep their weights, with no renormalisation. This works for Qwen3.5-MoE and Lfm2Moe. Custom MoE code is detected only if it follows the same fused-experts convention; otherwise expert parts are rejected with a clear message.
- **`block`:** the layer's output is replaced by its input, i.e. a residual identity.

All hooks are registered into a caller-held list and removed in `finally`.

## The light gate (tier `screened`)
A part passes only if all of these hold:
- The target skill drops at least 15% (relative) on both the search half and the hidden held-out half. The agent only sees pass/fail for the held-out half.
- Let the limit be max(5%, target drop / 3).
  - Every other active skill may drop by at most the limit on the search half. Rises are allowed, because relative changes on low-probability skills are noisy.
  - The target's own matched control may move by at most ± the limit, in either direction. A big rise there means the part only shifts which tokens get predicted.

  This is `gate_version` 2. `mine/regate.py` re-applies it to stored evidence.
- The target drop beats each of 5 random parts of the same size and kind in the same layer. For blocks, the random parts have the same length but a different start.

Optional tier `confirmed` (math parts only): greedy GSM8K accuracy on 40 problems drops at least 20% relative.

## Outputs per model (no merge conflicts)
- `catalog/<model>.jsonl`
- `parts/<model>/*.pt`, each a `{kind, model, layer, indices}` dict; blocks also store `layers`
- `mine/skills/<model>/<name>.json`, the frozen agent tests
- `logs/<model>/`, which holds:
  - `state.json`, used for resuming
  - the run transcripts
  - `experiments.jsonl`
  - `costs.jsonl`
  - `notes.jsonl`
