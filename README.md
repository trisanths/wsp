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

One graft against fine-tuning across 15 benchmarks. Host Llama-3.2-1B,
Qwen2.5-14B donor; both arms train on the same mixed-domain data, and none
of these benchmarks is trained on:

| Benchmark | Base | LoRA | Graft |
|---|---|---|---|
| MMLU | 44.6 | 48.0 | **59.4** |
| MMLU medical | 52.6 | 52.8 | **65.0** |
| MMLU math | 28.0 | 33.6 | **50.8** |
| MMLU science | 38.2 | 42.0 | **55.2** |
| MMLU computer science | 37.4 | 44.7 | **55.7** |
| MMLU humanities | 53.4 | 56.6 | **71.8** |
| MedMCQA | 41.4 | 41.6 | **47.2** |
| ARC-Challenge | 52.6 | 60.0 | **82.2** |
| ARC-Easy | 73.0 | 75.4 | **91.4** |
| HellaSwag | 34.6 | 45.0 | **65.0** |
| MMLU in Chinese | 40.0 | 38.0 | **49.8** |
| MMLU in Spanish | 37.8 | 38.0 | **52.8** |
| MMLU in Hindi | 31.6 | 35.8 | **39.0** |
| MMLU in Arabic | 37.8 | 35.4 | **42.8** |
| MMLU in Japanese | 36.4 | 35.8 | **43.2** |

The gain comes from the donor's weights. The same graft with the donor
block's weights scrambled scores 47.4 on MMLU, about the same as LoRA. The
real Qwen2.5-7B block scores 53.5.

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
