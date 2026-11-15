# dissect: propose -> ablate -> validate -> catalog

First pair: donor `Qwen/Qwen2.5-Math-1.5B`, task GSM8K. GPU work runs on Modal (app `swarmy-dissect`, L4).
The agent's brain is OpenAI (planner `gpt-6-astra`, worker `gpt-5.4-mini`).

## Files
- `prepare_data.py`: builds the fixed task slices with seed 0. `prepare_modal.py` runs it on Modal and writes `data/tasks.json`.
- `dissect_app.py`: the Modal class `Dissector`. It loads the model once and has methods for scoring, ablation hooks, random baselines, activation profile, attribution scan, validation evidence and index-file bytes.
- `client.py`: local side. It holds the validation gate (thresholds), writes `results/validations.jsonl`, and writes catalog records and `parts/<model>/<id>.pt` index files.
- `cli.py`: manual steps (`base`, `sanity`, `propose`, `ablate`, `validate`).
- `agent.py`: the OpenAI tool-calling loop. It writes `runs/<run_id>.jsonl` and appends lessons to `notes.jsonl`.

## Run
```
cd ~/swarmy
.venv/bin/modal run dissect/prepare_modal.py      # only if tasks need rebuilding
.venv/bin/modal deploy dissect/dissect_app.py     # after any change to dissect_app.py or data/
.venv/bin/python dissect/cli.py base
.venv/bin/python dissect/cli.py sanity
.venv/bin/python dissect/cli.py propose
.venv/bin/python dissect/cli.py validate '{"kind":"attn_heads","layer":22,"indices":[6,9,10,11]}' math-reasoning-heads
.venv/bin/python dissect/agent.py 30
```
After a redeploy, stop warm containers, or old code keeps answering for up to 4 minutes:
`modal container list` then `modal container stop -y <id>`.

## Tasks and metrics
- **gsm8k / gsm8k_heldout:** geometric-mean probability of the digit tokens in the gold worked solution (2-shot), on 200 / 100 problems. A fast proxy only.
- **gsm8k_gen / gsm8k_gen_heldout:** greedy-generation exact-match accuracy on 60 problems from each split. **This is the gated target.**
- **trivia:** probability of TriviaQA answers (3-shot). **wiki:** WikiText token probability. **wiki_digits:** probability of digit tokens in WikiText (numbers without arithmetic). These are the controls.

## Gate (client.py)
A part passes only if all of these hold:
- the generation-accuracy drop is at least 0.30 under both zero and mean ablation, on both the score and held-out subsets
- every control drop is below 0.10 under both methods
- the generation drop is larger than every one of 10 random same-size parts in the same layer (percentile at least 0.95)

Passing parts go to `../catalog.jsonl`, with index files in `../parts/qwenmath1.5b/`. Every result, pass or fail, goes to `results/validations.jsonl`.

## Part kinds
- **mlp_cluster:** neuron ids at the input of one layer's `down_proj`.
- **attn_heads:** query-head ids, ablated as that head's 128-wide slice of the `o_proj` input. This is correct under GQA.
- **block:** a contiguous range of layers, skipped (zero) or replaced by the layer's mean residual update (mean).
