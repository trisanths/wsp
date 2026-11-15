# Sapien: weight-space programming

Sapien builds one model out of parts of other models. You take a small
model and wire a frozen block of layers from a bigger model into it, even
when the two come from different model families with different tokenizers.
Only small connectors are trained. The result is one network: the block's
weights are copied in, and the bigger model never runs.

## How it works

```
host layers 1..7 ──► learned mix ──► encoder ──► donor block (frozen, e.g. Qwen2.5-14B layers 7..34)
                                                        │
host layer 7 ◄──── + gate × decoder ◄───────────────────┘
   │
   ▼
host layers 8..16 ──► answer
```

1. **Pick a donor block.** This is a run of layers from a bigger model, such
   as Qwen2.5-14B or a math or medical specialist. It stays frozen.
2. **Attach it beside the host.** The host is a small model such as
   Llama-3.2-1B. An encoder reads a learned mix of the host's hidden states
   and maps it into the donor's space. A decoder maps the block's output
   back. A gate adds it to the host. The gate starts at zero, so a freshly
   attached graft is exactly the original model.
3. **Align.** The encoder learns to give the block inputs that look like
   the donor's own hidden states.
4. **Integrate.** The encoder, decoder and gate train on multiple-choice
   data. None of it comes from the benchmarks we report
   (`wsp/healdata.py` checks the sources).

LoRA, the fine-tuning baseline, gets the same trainable-parameter budget.

## Results

Host Llama-3.2-1B, trained on each domain and scored on that domain:

| Domain | Base | LoRA | Adapter only | Graft |
|---|---|---|---|---|
| Math (AQuA) | 25.5 | 34.5 | 39.0 | **52.0** |
| Medicine (HeadQA) | 45.8 | 51.7 | 42.8 | **54.8** |
| Medicine (MedMCQA) | 41.4 | 39.7 | 37.8 | **45.1** |
| Science (SciQ) | 85.8 | 87.5 | 88.2 | **89.8** |
| MMLU | 44.6 | 47.1 | 45.6 | **53.1** |

"Adapter only" is the same encoder and decoder with no donor block. The
donors are Qwen2.5-Math-7B for math, Apollo2-7B for medicine, and
Qwen2.5-7B for science and MMLU. MedMCQA is a separate benchmark that we
never train on.

Bigger donors give bigger gains (MMLU held-out):

| Host | Base | LoRA | 7B donor | 14B donor | 32B donor |
|---|---|---|---|---|---|
| Llama-3.2-1B | 44.6 | 46.0 | 53.6 | 59.8 | **62.6** |
| Llama-3.2-3B | 57.4 | 60.7 | — | 66.0 | **66.3** |

The full log of experiments is in
[docs/research/2026-11-15-parallel-graft.md](docs/research/2026-11-15-parallel-graft.md).

## Run it

CPU tests (tiny models, no downloads):

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]" && .venv/bin/pytest -q
```

Real runs use [Modal](https://modal.com). One batch of experiments:

```bash
.venv/bin/modal run --detach modal_hill.py --batch h4
.venv/bin/python scripts/hill_table.py h4      # fetch and print the results table
```

## Where things are

- `wsp/pgraft.py`: the graft (encoder, donor block, decoder, gate),
  alignment and training.
- `modal_hill.py`: the experiments, data and evals.
- `wsp/healdata.py`: training data sources and the leak check.
- `wsp/`: the Sapien language and compiler. See
  [docs/quickstart.md](docs/quickstart.md) and
  [docs/language.md](docs/language.md).
- `wsp_eval/`: the eval harness. `studio/`: the web app. `library/`: the
  parts library.
