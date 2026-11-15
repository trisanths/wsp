# wsp_eval: eval harness and extraction (Person B, part 1b)

Brief: [briefs/brief-B-eval-interp.md](../briefs/brief-B-eval-interp.md), including the
2026-11-14 mixed-family update at the end.

## Interface 4: `eval(model_path, split, tasks) -> scores`

```python
from wsp_eval import eval, set_limit

set_limit("swarm", 200)                     # max search-split eval() calls for this method
scores = eval("outputs/my_graft", "search", ["gsm8k"], namespace="swarm")
scores["tasks"]["gsm8k"]["score"]           # 0.0 to 1.0
scores["combined"]                          # normalized combined score, once a reference exists
scores["fit_r2"]                            # R squared per graft from compile_report.json
scores["label"]                             # "SEARCH SPLIT - for search only, not a result"
```

- `model_path`: a model directory (plain or compiled by `wsp`), a Hugging Face id, or `fake:<name>`
  (deterministic fake outputs, for wiring tests).
- `split`: `search` or `heldout`. `heal` is training data and is refused.
- Backends: vLLM for plain models on CUDA; Hugging Face `generate` for grafted models
  (`program.json` lists grafts, loaded with `wsp.load`), for ablation probes, and on laptops.
  Override with `WSP_BACKEND=vllm|hf|fake`. Greedy decoding, fixed seed, the same prompt for every model.
- Cache: SQLite, keyed by (model hash, backend, max tokens, rendered prompt). Shared across processes.
- Budget: one counter file, one namespace per method (`swarm`, `evo_merge`, `lora`, `full_ft`, `interp`, ...).
  Every search-split call counts, cached or not. Compare methods only at equal `eval_calls`.
- Held-out: each (model, task) runs once; a second call raises `HeldoutAlreadyRun`.
- Every call appends a line to `experiments.jsonl` (timestamp, model, split, scores, eval count, fit R squared, seed).

State lives in `work/` (gitignored) under `WSP_ROOT` (default: the current directory).

## Splits

| Task | search | heal (training) | held-out |
|---|---|---|---|
| `gsm8k` | 200 items from GSM8K train (seeded) | rest of GSM8K train (7273) | GSM8K test (1319) |
| `mmlupro[:<category>]`, `bbeh[:<task>]`, `bbeh_mini`, `mmlu[:<subject>]` | 150 items of the test set (seeded) | none | rest of the test set |

The three GSM8K sets do not overlap (checked in tests). Heal data for the compiler:
`from wsp_eval.splits import heal_texts; heal_texts("gsm8k")`.

## Combined score

Mean over skills of `(score - base) / (best single-skill model - base)`, clipped to [0, 1]. Store the
reference once per split with `save_reference(split, base_scores, parent_scores)`
(`scripts/eval_first_numbers.py` does it). Skills whose best parent does not beat the base are listed in
`skipped`, not divided by zero.

## Claims: controls and seeds

```python
from wsp_eval import claims
verdict = claims.compare(graft_results, random_cluster_control_results, "gsm8k", lora_results=lora_results)
print(claims.format_verdict(verdict))
```

Results are paired by `program_seed`. A gain counts only with at least 3 paired seeds, a positive mean gain
over the random-cluster control, and a win on every seed. The LoRA control is reported alongside.

## Extraction (interp)

```bash
python scripts/extract_donor.py --model Qwen/Qwen2.5-Math-1.5B --task gsm8k            # forward passes only
python scripts/extract_donor.py --model Qwen/Qwen2.5-Math-1.5B --task gsm8k --validate 2  # + ablation checks (GPU)
MODAL_PROFILE=trisanth modal run modal_eval.py::extract --model Qwen/Qwen3-30B-A3B --gpu H100 --kinds expert
```

For each layer, units are scored by how much more they respond to GSM8K training text (heal split, never
search or held-out) than to WikiText-2:

- `mlp_cluster`: Cohen's d of |neuron activation| (the input to `down_proj`)
- `attn_heads`: Cohen's d of each query head's output norm (its slice of the `o_proj` input)
- `expert`: log2 ratio of how often the router picks each expert

The top units of the best layers are written as `.pt` index files, the format in the compiler design:
`{kind, model, layer, indices: LongTensor, signature}`. Each comes with three random same-size controls from the
same layer (`*_control_seed{0,1,2}.pt`), which are the required main control. `records.jsonl` holds catalog
records for Person A (`type`, `model`, `layer`, `units`, `controls`, `signature`).

`--validate N` checks function: knock the units out of the donor (hooks), measure the GSM8K search-split drop,
and compare with knocking out the random control. `specific: true` means the drop beats the random drop by more
than two standard errors. Every step is logged to `interp_log.jsonl` as hypothesis, action, result.

## Same-base mask tools (still work)

`probe(component, layers)` (built with `wsp` compile), `overlap(a, b, by="layer")`, the mask `InterpAgent`
(`scripts/run_interp.py`), and heatmaps (`scripts/make_heatmaps.py`).

## Tests

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,compile,eval]" "transformers==4.57.1" "datasets==4.1.1"
.venv/bin/pytest -q
```

`transformers==4.57.1` matches the Modal image; `datasets==4.1.1` is the newest release that works with the
`huggingface_hub` that version needs.

## Hour-8 gate, seed 0 (2026-11-15). SEARCH SPLIT: an interim check, not a result.

Host unsloth/Llama-3.2-1B-Instruct, donor Qwen/Qwen2.5-Math-1.5B, GSM8K search split (200 items from
train, same items as `wsp.data.gsm8k_splits`), scored by `eval()` with the hf backend for every row.
Part: 512 MLP neurons from donor layer 10, chosen by contrast (GSM8K heal text vs WikiText-2), grafted at
host layer 6 with rank-64 glue. Heal for every arm: 300 steps, gsm8k_train, lr 3e-4, batch 8, seq 384.
Run with `modal run modal_eval.py::gate --seeds "0"`. Raw: `results/gate_hour8_s0.json`.

| Model | GSM8K search | fit R2 (pooled / per dim) |
|---|---|---|
| Host | 0.230 | |
| Donor | 0.450 | |
| Graft + heal | 0.415 | 0.68 / 0.53 |
| Control 1: random 512 neurons, same layer, same glue and heal | 0.385 | 0.68 / 0.53 |
| Control 2: host LoRA matched to glue + part params (2.82M) | 0.440 | |

Reading: the pipeline runs end to end. The graft's 0.03 lead over the random part is inside the noise
(paired standard error about 0.05 on 200 items), and the matched LoRA scores highest. One seed is not a
claim; `--seeds "0 1 2"` reuses the seed-0 compiles and adds the other two.
