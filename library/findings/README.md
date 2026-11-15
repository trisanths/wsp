# Findings

`findings.jsonl` holds one record per measured finding. The planner and
the agents read it before they write a program. Most records come from
`results/README.md` and the raw rows in `results/`; `scripts/library.py
build` writes them, and people and agents append more with
`scripts/library.py add`.

A record has the common fields (`id`, `kind: "finding"`, `title`,
`status`, `added_by`, `source`, `updated`, `skills`, `models`, `evidence`,
`generated`) plus:

- `claim`: the finding in one or two plain sentences, with its numbers;
- `setup`: the run, the hosts and donors, the steps, and the seeds;
- `numbers`: the measured values, as accuracies in percent or losses in
  nats, keyed by arm, as printed in `numbers_source` (usually
  `results/README.md`, which rounds to one decimal place);
- `numbers_source`: the file the `numbers` come from. The evidence
  `value` and `control_value` are exact from the raw file in `source`;
- `status_reason`, for a rejected finding.

Status:

- `validated`: measured on held-out items or held-out texts;
- `candidate`: measured on a search or dev split, or with one seed;
- `rejected`: a claim that a later run showed to be wrong, kept so nobody
  repeats it (for example `f_run1_fit_r2_inflated`).

The main findings so far:

- Taps carry knowledge into small hosts: held-out MMLU 68.3 for Llama-1B
  with a Qwen2.5-7B tap, against 47.7 for a LoRA (`f_run7_heldout_tap`).
- Taps carry no multi-step reasoning (`f_run5_taps_no_reasoning`) and hurt
  free-form answers (`f_run7_tap_hurts_generation`).
- Feed the donor its attention sink at position 0
  (`f_audit_attention_sink`), and train the stitch on the donor's own LM
  loss (`f_stitch_donor_lm`).
- Single-layer clusters and in-stream blocks do not beat a random part of
  the same size (`f_run2_selection_equals_random`,
  `f_stitch_graft_ties_random`, `f_part_gate_search_not_heldout`).
- A donor block written into a 0.5B base with a near-lossless stitch still
  does not beat a random block on held-out MMLU, because the base's later
  layers cannot read it (`f_run14_reader_needed`).
- Carving more than half the layers destroys a model
  (`f_run15_carving_half_destroys`).
- Models are right on different items (oracle 96.4% against 79.3% for the
  best model on held-out MMLU), but averaging captures little of it
  (`f_round10_complementarity`).

Query them:

```bash
.venv/bin/python scripts/library.py query --kind finding
.venv/bin/python scripts/library.py query --kind finding --skill tap --status validated
```
