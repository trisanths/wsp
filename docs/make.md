# Making a model from a request

You write down what the model is for and what it may cost. wsp plans a
few ways to build it from open models that exist now, builds and scores
each one, and gives you the best one with a report card. Nothing is
pretrained. Each candidate builds in minutes on one GPU (a tap with a
7B donor takes about 100 seconds on an H100, loads included; bigger
donors take longer), and the candidates run in parallel.

The code is in `wsp/make.py` (the winner and the report card),
`wsp/planner.py` (the plan), `wsp/profiles.py` and `profiles.json` (what
is known about each model), and `modal_make.py` (the build on Modal).

## The request

A request is a JSON file in `requests/`:

```json
{
  "name": "med_mcq_1b",
  "description": "Medical multiple choice (MedMCQA) with at most 1B parameters at decode, as accurate as possible.",
  "task": {"kind": "mcq", "benchmark": "medmcqa", "subjects": null},
  "constraints": {"max_decode_params_b": 1.0, "max_total_params_b": null,
                  "max_train_minutes": null, "gpu": "H200"},
  "objective": "max_accuracy",
  "accuracy_target": null
}
```

- `task.kind` is `mcq` (pick a letter) or `generate` (write an answer,
  such as GSM8K). `task.benchmark` names the test set, and
  `task.subjects` can limit it, for example to the MMLU law subjects.
- `max_decode_params_b` limits the parameters that run for each new
  token, which sets the serving speed. `max_total_params_b` limits what
  must sit in memory: the host plus every tap donor. `max_train_minutes`
  limits the build time. A null means no limit.
- `objective` is `max_accuracy` (the most accurate model that fits) or
  `min_cost_at_target` (the cheapest model that reaches
  `accuracy_target`). Accuracies are fractions, so 40% is `0.40`.

`python -m wsp.make <file> --dry-run` checks the file against this
schema and prints the plan.

## From request to model

1. The planner reads the request and the model profiles. A profile has
   each model's size, depth, tokenizer family, and measured scores
   (round 10 in `results/README.md`). The planner uses rules from the
   measured runs:
   - For knowledge questions, a small host reads a big donor through a
     tap. The donor is read near 70% of its depth and the host writes
     near 40% of its depth. In run 7, Llama-1B went from 44.7 to 68.3
     on held-out MMLU with a Qwen-7B tap (the donor alone scores 70.9),
     and in run 8 a Qwen-0.5B host with a Qwen-14B tap scored 74.8.
   - A tap does not help a host that is already as strong as the donor
     (run 9), does not carry multi-step reasoning (GSM8K, run 5), and
     hurts short free-form answers (TriviaQA, run 7).
   - For reasoning tasks such as GSM8K, the bar is a LoRA (rank 64, lr
     5e-5 gave 47.2 on Llama-1B). A LoRA cannot add knowledge (MMLU 47.7
     against 68.3 for the tap).
   - A bake candidate distills a tapped host into a plain host, so no
     donor runs at inference.
2. The planner returns up to `k` candidates (6 by default). Each has a
   program, a prediction (accuracy, decode parameters, prefill
   parameters, build minutes), and a rationale.
3. `modal_make.py` runs each candidate in its own container. It compiles
   the program with `wsp.compiler`, bakes it when the candidate asks for
   that, loads it with `wsp.load`, scores it on held-out items of the
   benchmark, and times decoding at batch 1. By default it also runs
   each planned host alone, with no training, so the card has measured
   host numbers on the same items.
4. `wsp.make` picks the winner. Under `max_accuracy`, it is the most
   accurate candidate that ran without an error and fits every limit.
   Under `min_cost_at_target`, it is the candidate with the fewest
   decode parameters among those that reach the target, then the fewest
   total parameters, then the shortest build. When none reaches the
   target, the card names the most accurate one and says the target was
   missed.
5. The report card lists the winner (its program text, accuracy, decode
   and prefill parameters, decode speed, build time, and model folder),
   every candidate with its predicted and measured accuracy, the
   baselines, and notes in plain English. The baselines are the host
   alone, the best single model in the total budget and in the decode
   budget (measured when the plan ran it, else from its profile), and a
   LoRA on the same host when the plan had one.

## Worked example

The request above asks for a medical model that decodes with at most 1B
parameters. The dry run:

```bash
.venv/bin/python -m wsp.make requests/med_mcq_1b.json --dry-run
```

prints the plan. Its first single-tap candidate is this program:

```wsp
host Qwen/Qwen2.5-0.5B-Instruct
seed 0
tap Qwen/Qwen2.5-32B-Instruct layer 45 at 10 id qwen32b
tap_train data medmcqa_train steps 500 batch_size 8 seq_len 256 branch_lr 0.0003 gate_lr 0.03
```

The planner chose Qwen2.5-0.5B as the host, because it fits the 1B
decode limit. It chose Qwen2.5-32B as the donor, because it has the best
MedMCQA score in the profiles (67.2 in round 10) and recovery grows with
donor size. It reads the donor at layer 45 of 64 (about 70% depth) and
writes into host layer 10 (about 40% depth). The prediction is 64.2%
with 0.50B decode parameters and about 6 minutes of build time. The same
plan has a staged variant (tap first, then a rank-16 LoRA), a two-tap
variant, and the same designs on the smaller LFM2-350M host.

To build and score the plan:

```bash
MODAL_PROFILE=trisanth .venv/bin/modal run modal_make.py --request requests/med_mcq_1b.json
```

The candidates run at the same time, one H200 each. The run writes the
plan to `results/make/med_mcq_1b.plan.json`, every measured result with
its per-item records to `results/make_med_mcq_1b_shard0.json`, and the
report card to `results/make/med_mcq_1b.json`, and prints the card. The
models are on the `wsp-models` volume under `/vol/make/med_mcq_1b/`.
The card then says, for example, how far the winner is above Qwen-0.5B
alone, and how far it is from Qwen-32B, the best model in the budget
when the total size has no limit.

## More options

- `--requests-dir requests` runs every request in the folder in one
  go. The results go to `results/make_all_shard0.json`, and each request
  gets its own card.
- `--eval-count N` scores N held-out items in place of the defaults
  (2000 for MMLU, 1500 for MedMCQA, all 1273 for MedQA, 2000 for
  MMLU-Pro, 500 for GSM8K).
- `--use-llm` asks the LLM planner. Its plan can change between runs, so
  save one first with `python -m wsp.make <file> --llm --dry-run
  --save-plan plan.json` and pass `--plan-file plan.json`.
- To spread one request over several Modal workspaces:

  ```bash
  python scripts/modal_fanout.py modal_make.py make_med_mcq_1b --profiles a,b,c \
      --extra "--request requests/med_mcq_1b.json"
  python -m wsp.make requests/med_mcq_1b.json --results results/make_med_mcq_1b_rows.json
  ```

  The second command builds the card from the merged shards and writes
  `results/make/med_mcq_1b.json`.

## Limits

- Held-out scores are on items the planner's numbers did not come from
  (MMLU test positions 1000 on, MedMCQA validation positions 1000 on),
  but each request still uses one sample of each benchmark.
- MedQA and MMLU-Pro have no profile scores yet, so the card's
  source-model baselines for them come from the measured host-alone
  runs only.
- A LoRA without taps trains on the full text of its pairs (prompt and
  answer), since `wsp.data` holds only GSM8K texts. With taps, training
  covers only the answer.
- A GSM8K model that trained on `gsm8k_train` is prompted in that raw
  format. Other models are prompted with their chat template.
