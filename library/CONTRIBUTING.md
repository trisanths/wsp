# Contributing to the Sapien library

These are the steps and rules for adding or updating an entry. They apply
to people and to agents. `library/README.md` describes the layout and the
fields.

## Add an entry

1. Choose the kind: `model`, `deletion_mask`, `mlp_cluster`, `attn_heads`,
   `expert`, `block`, `recipe`, `finding`, or `stitch`.
2. Write one JSON file for the entry. One entry goes in one file. (Findings
   and stitches are lines in `findings/findings.jsonl` and
   `stitches/stitches.jsonl`; `add` appends the line for you.)
3. Fill the required fields:
   - every entry: `id`, `kind`, `title`, `status`, `added_by`, `source`,
     `skills`, `models`, `evidence`;
   - a model: `hf_id`, `family`, `params_b`, `layers`, `hidden`, `vocab`,
     `tokenizer_family`, `license`, `scores`, `structure_notes`,
     `components`;
   - a component: `model`, `layer`, `layers`, `units_file`,
     `units_remote`, `signature`;
   - a recipe: `host`, `program`, `text`, `program_file`, `syntax_gap`;
   - a finding: `claim`, `setup`, `numbers`;
   - a stitch: `host`, `donor`, `host_layers`, `donor_layer`,
     `stitch_kind`, `objective`, `steps`, `sink`, `penalty`.
   Use `null` for a value you do not know. Do not guess.
4. Give provenance:
   - `added_by`: a person or agent name, for example `Hanav (Person A)` or
     `agent: atlas-miner-2`;
   - `source`: the run or file the entry comes from, for example
     `results/round7_rows.json (run 7)` or `person_a/catalog/qwenmath1.5b.jsonl`.
5. For a component with an index file, set `units_file` to the `.pt` path:
   an absolute path, a path relative to the folder of your JSON file, or a
   path relative to the repo root. `add` copies the file to
   `components/<org>__<name>/` and rewrites the path. Do the same for
   random controls in `controls` (a list of paths).
6. Run from the repo root:

   ```bash
   .venv/bin/python scripts/library.py add my_entry.json
   ```

   `add` refuses an id that exists, checks the entry, writes it, and
   rewrites `index.jsonl`. It sets `"generated": false`, so `build` never
   changes the entry, and it puts the entry's `added_by` on evidence items
   that have none. If `add` prints errors, fix the file and run it again;
   nothing is written until the entry is valid.

A minimal component entry (a candidate, so it needs no evidence yet):

```json
{"id": "mlp_cluster_phi4_l3_gsm8k_agentx", "kind": "mlp_cluster",
 "title": "Phi-4 layer 3, 64 MLP neurons by GSM8K contrast",
 "status": "candidate", "added_by": "agent: atlas-miner-2",
 "source": "results/my_extract_rows.json",
 "skills": ["math", "gsm8k"], "models": ["microsoft/phi-4"], "evidence": [],
 "model": "microsoft/phi-4", "layer": 3, "layers": [3],
 "units_file": "my_parts/phi4_l3.pt", "units_remote": null,
 "controls": ["my_parts/phi4_l3_control_seed0.pt"],
 "signature": {"method": "activation contrast, Cohen's d", "n_units": 64}}
```

## Remove an entry you added by mistake

There is no remove command, because entries are shared. To remove your
own entry before anyone uses it, delete its JSON file (or its line in
`findings/findings.jsonl` or `stitches/stitches.jsonl`) and the index
files that `add` copied for it, then run
`.venv/bin/python scripts/library.py index` and `validate`. Never remove
another contributor's entry; mark it `rejected` with evidence instead.

## Add evidence to an entry that exists

Never overwrite another contributor's evidence. Append to its evidence
list:

```json
{"id": "mlp_cluster_phi4_l17_gsm8k",
 "evidence": [{"claim": "graft at Llama-1B layer 6 vs a random same-size cluster",
               "metric": "GSM8K accuracy, mean over seeds",
               "value": 0.33, "control_value": 0.31,
               "control": "random same-size cluster, same layer",
               "split": "GSM8K test, 1319 items (held-out)", "seeds": [0, 1, 2],
               "source": "results/final_heldout_microsoft__phi-4_l17.json",
               "added_by": "agent: gate-runner-1"}]}
```

```bash
.venv/bin/python scripts/library.py add evidence.json --append-evidence
```

To change the status as well, add `"status"` and `"status_reason"` to the
file. Change it only when the new evidence supports the change. `build`
keeps hand-added evidence on a generated entry, and then it keeps the
entry's status too.

## Evidence rules

Each evidence item needs `claim`, `metric`, `value`, `control_value`,
`split`, `seeds`, and `source` (use `null` for a value that does not
apply), and `added_by` when a person or agent adds it. Give accuracies as
fractions from 0 to 1 and losses in nats. Add `"control"` to say what the
control was. Add `"lower_is_better": true` when a lower value is better
(a loss or a penalty).

- `validated` needs evidence:
  - a component or a recipe: a measured gain over a random same-size
    control (for a recipe, over its baseline) on held-out data. The item
    has `value`, `control_value`, `seeds`, and a `split` with "held-out" in
    its name, and `value` beats `control_value`. For a component, the
    item's `control` text names the random part (it must contain
    "random").
  - a model: measured scores, each with `value`, `split`, `prompt`, and
    `source`. Name the split.
  - a finding or a stitch: a measured `value` with its `split`.
- A gain on a search or dev split keeps the entry a `candidate`.
- Ablation inside the donor and contrast effect sizes keep a part a
  `candidate`. They do not show that the part helps a host.
- `rejected`: the entry was measured and failed. Keep it. Negative results
  are data. Give the evidence and a `status_reason`.
- Every number names its source file. Never invent a number, a name, or a
  date.

## Before you commit

```bash
.venv/bin/python scripts/library.py validate
```

It checks the required fields, that ids are unique, that every named file
exists inside the library, that statuses and kinds are legal, that
`validated` entries have the evidence above (including the gain over the
control), and that `index.jsonl` is current. Fix every error. If it
says the index is out of date, run `scripts/library.py index`.

## Do not

- Do not edit or delete another contributor's entry or evidence item.
- Do not change a generated entry's generated fields by hand; the next
  `build` rewrites them. Add evidence with `--append-evidence`, or copy
  the entry to a new id with `"generated": false`.
- Do not move or delete the files of Person A's export
  (`library/catalog.jsonl`, `library/<model>/`).
- Do not commit model weights. Index files and masks are small; weights
  stay on Hugging Face.
