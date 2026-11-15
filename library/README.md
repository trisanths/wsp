# Sapien library

The library holds the mined building blocks that Sapien programs are
written from. Sapien writes a new model from parts of open models: one
network, and the only training is the stitch and the heal. Before a person
or an agent writes a program, they look here for three things: the parts
that exist, the measured evidence for each part, and what earlier runs
found.

Several teams' people and agents add to this library. Every entry says who
added it, where it came from, and what was measured. Negative results stay
in the library as `rejected` entries, because they tell the next person
what not to try.

## Layout

| Path | What it holds | One entry is |
|---|---|---|
| `index.jsonl` | one line per entry: id, kind, path, title, skills, models, status, added_by, updated | a line |
| `models/<org>__<name>.json` | model profiles: size, layers, hidden size, vocabulary, license, measured scores, structure notes, and links to the model's components | a file |
| `components/<org>__<name>/<id>.json` | mined components of that model, with the small `.pt` index file next to the JSON when one exists | a file |
| `findings/findings.jsonl` | measured findings from `results/README.md`, with numbers, setup, and the source file (see `findings/README.md`) | a line |
| `stitches/stitches.jsonl` | measured stitch recipes: host, donor, layers, kind, objective, steps, sink, and the penalty before and after | a line |
| `recipes/<id>.json` | programs that were run, with the measured outcome; `recipes/<id>.wsp` holds the text form when the syntax can write it | a file |
| `catalog.jsonl`, `qwenmath1.5b/` | Person A's export of passing mined parts, ready for the compiler (see the last section) | a line |

Component kinds:

- `deletion_mask`: a Deletion Tuning mask over a base model (the 37 masks in
  the root `catalog.jsonl`, stored on Hugging Face).
- `mlp_cluster`: a set of MLP neurons from one donor layer.
- `attn_heads`: a set of query heads from one donor layer.
- `expert`: one or more MoE experts from one donor layer. None is mined yet.
- `block`: whole donor layers. A block can be an in-stream graft
  (`qmath_block_8_19`) or a tap read point (the donor runs layers 0..L and
  the host reads layer L).

To use a component in a program, use its `catalog_id`. For a part with an
index file, the root `catalog.jsonl`, `parts_extracted/*/records.jsonl`,
or this folder's `catalog.jsonl` already holds the compiler record.

## Entry schema

Every entry has these fields:

| Field | Meaning |
|---|---|
| `id` | unique in the whole library; letters, digits, `_`, `.`, `-` |
| `kind` | `model`, `deletion_mask`, `mlp_cluster`, `attn_heads`, `expert`, `block`, `recipe`, `finding`, or `stitch` |
| `title` | one plain line |
| `status` | `candidate`, `validated`, or `rejected` |
| `status_reason` | why the entry has its status (needed for `rejected` without evidence) |
| `added_by` | the person or agent that added it, for example `Hanav (Person A)` or `agent: swarm-planner-3` |
| `source` | the run or file it came from, for example `results/round7_rows.json (run 7)` |
| `updated` | date of the last change, `YYYY-MM-DD` |
| `skills` | short lowercase tags, for example `math`, `gsm8k`, `knowledge`, `mmlu`, `stitch` |
| `models` | Hugging Face ids the entry is about |
| `evidence` | a list of evidence items (below) |
| `generated` | `true` when `scripts/library.py build` made it; `false` for an entry a person or agent added |

Each kind adds fields:

- model: `hf_id`, `family`, `params_b`, `active_params_b`, `layers`, `hidden`,
  `vocab`, `tokenizer_family`, `license`, `experts`, `needs`, `scores`,
  `structure_notes`, `components`. A score is
  `{"value", "split", "prompt", "source"}`, and the value is a fraction
  from 0 to 1.
- component: `model`, `layer` (for one-layer parts), `layers`,
  `units_file` (a path inside the library, or null), `units_remote` (where
  the index file or mask lives outside the repo), `signature` (method,
  data, and effect sizes as recorded), and often `controls`, `catalog_id`.
- recipe: `host`, `program` (the program JSON), `text` (the program in the
  text syntax, or null), `program_file`, `syntax_gap` (what the syntax
  cannot say, or null), `components`.
- finding: `claim`, `setup`, `numbers`, and `numbers_source` (the file
  that prints the `numbers`, often `results/README.md`, which rounds to
  one decimal place). The evidence `value` and `control_value` of a
  generated finding are exact from the raw file in `source`.
- stitch: `host`, `donor`, `host_layers`, `donor_layer`, `stitch_kind`,
  `objective`, `steps`, `sink`, `penalty`.

An evidence item is:

```json
{"claim": "graft vs a random same-size cluster through the same glue and heal",
 "metric": "GSM8K accuracy, mean over seeds",
 "value": 0.3139, "control_value": 0.3199,
 "control": "random same-size cluster, same layer",
 "split": "GSM8K test, 1319 items (held-out)",
 "seeds": [0, 1, 2],
 "source": "results/final_heldout_microsoft__phi-4.json",
 "added_by": "Person B"}
```

`control` says what the control was. It is optional, except on the item
that makes a component `validated` (see below). `note` is optional. Values
are fractions from 0 to 1 for accuracies, and nats for losses and
penalties. Higher is better unless the item has `"lower_is_better": true`
(set it for a loss or a penalty). Items that the build made carry
`"generated": true`. Items that a person or agent added need `added_by`.

## Evidence rules

- `validated` for a component or a recipe: at least one evidence item
  measures a gain over a control on held-out data. The item has `value`,
  `control_value`, `seeds`, and a `split` with "held-out" in its name, and
  `value` beats `control_value`. For a component, the control is a random
  part of the same size from the same layer, and the item's `control` text
  says "random". `validate` checks all of this.
- `validated` for a model: measured scores, each with its split, prompt,
  and source.
- `validated` for a finding or a stitch: a measured value with its split.
- `candidate`: not measured yet, or measured only on a search or dev split.
  Most entries are candidates. A gain on the GSM8K search split is not
  enough: the Phi-4 layer 16 cluster gained 2.8 points there and lost 0.6
  held out (`f_part_gate_search_not_heldout`).
- `rejected`: measured, and it failed. Keep the entry and its evidence.
- Ablation in the donor (Person A's mining gate) shows that a part matters
  inside its own model. It does not show that the part helps a host, so
  those parts stay `candidate` until a graft beats its random control.
- Contrast effect sizes (Person B's extraction) select parts. They are not
  evidence of function.

## Who added what

| `added_by` | Work | Where it comes from |
|---|---|---|
| `Trisanth` | the compiler and the runs in `results/README.md` (runs 1 to 9, 14, 15, the audit, the baselines, the stitch study) and round 10 | `results/README.md`, `results/round*_rows.json`, `sweep2_rows.json`, `graft_run1_s0.json`, `audit_sink_rows.json`, `baselines_rows.json`, `stitch_study_rows.json`; the masks in the root `catalog.jsonl` |
| `Hanav (Person A)` | mining in the donor, with ablation gates | `person_a/catalog.jsonl`, `person_a/catalog/*.jsonl`, `person_a/parts/` |
| `Person B` | part 1b: the eval harness, contrast extraction with random controls, the part gates, the held-out run, the bake, the stitch graft, the exit stitch, and the mask signatures | `wsp_eval/`, `parts_extracted/`, `results/gate_hour8_*.json`, `part_gate_*.json`, `final_heldout_*.json`, `base_scores_heldout.json`, `bake_rows.json`, `stitch_graft_s0.json`, `exit_stitch_s0.json` (git author `airjan-airlines`, branch `person-b-eval`) |
| `Rex (part 3 pilot)` | the swarm's pilot parts | `data/pilot_catalog.jsonl`, `data/pilot_catalog_mined.jsonl` |

The model profiles carry `added_by: Trisanth`; each score names its own
source file.

## How to contribute

For people:

1. Write one JSON file for the new entry, with the fields above and
   `"generated": false`. Put a part's `.pt` index file next to it.
2. Run `python scripts/library.py add my_entry.json`. It checks the entry,
   copies a `units_file` or `controls` that live outside the library,
   writes the entry to its place, and rewrites `index.jsonl`.
3. To add a measurement to an entry that exists, put
   `{"id": ..., "evidence": [...]}` in a file and run
   `python scripts/library.py add file.json --append-evidence`. Add
   `"status"` and `"status_reason"` to the file to change the status, and
   only when the new evidence supports the change.
4. Run `python scripts/library.py validate` before you commit.

For agents:

- Read `index.jsonl` first, then `findings/findings.jsonl`, then the
  entries you need. `python scripts/library.py query --kind block --skill
  knowledge` and `--status validated` filter the index.
- Add entries only through `scripts/library.py add`. Do not edit another
  contributor's entry by hand, and never remove or change an evidence item.
- Put your agent name in `added_by` on the entry and on each evidence item.
- Never invent a number. Every value names its source file.
- Run `python scripts/library.py validate` and fix every error before you
  commit.

The full rules are in `CONTRIBUTING.md`.

## Skill-map entries (`map_block_*`, `map_expert_*`)

`person_a/mine/maps.py` measures one model at a time. It skips each decoder layer and records how much
the loss rises on a skill's text, minus the rise on general text (nats). On MoE layers it counts how
often the router picks each expert on the skill's text and on general text. The text is plain
non-benchmark text, and its description is stored in `selection_data` on each entry.
`scripts/library_from_maps.py` turns the map files in `atlas/pool_entries/` into entries:

- `block`: one per model, skill, and block (a run of layers chosen on the search half). `layers` lists them.
- `expert`: one per model, skill, and MoE layer. The `.pt` file holds `{kind, model, layer, indices}`.
- Each entry has `params_b`, `n_layers`, `hidden`, `family`, `architecture`, `license`, and `revision` at top level.
- All are `candidate`. The numbers come from the donor alone and say where a skill lives, not that a part helps a host.
- Blocks need `split_half_r` of at least 0.5 and a held-out `check_mass` of at least 0.2. Expert layers need 2 experts.

```bash
.venv/bin/python scripts/library_from_maps.py write --entries-dir atlas/pool_entries   # safe to rerun
.venv/bin/python scripts/library.py query --kind block --skill law --model qwen
.venv/bin/python scripts/library_from_maps.py find --kind block --skill law --max-params 8
.venv/bin/python scripts/library_from_maps.py find --kind expert --skill code --max-hidden 3072 --license apache
```

`find` also takes `--family`, `--min-params`, and `--status`. Entry ids start with `map_`.

## Regenerating the generated entries

```bash
.venv/bin/python scripts/library.py build      # read the repo sources, write generated entries
.venv/bin/python scripts/library.py index      # write index.jsonl
.venv/bin/python scripts/library.py validate   # exit 1 on any error
.venv/bin/python scripts/library.py query --kind mlp_cluster --status rejected
```

`build` reads `catalog.jsonl`, `results/*.json`, `parts_extracted/`,
`person_a/`, `data/pilot_catalog*.jsonl`, `profiles.json`,
`atlas/zoo.json`, `person_a/registry/`, and `docs/donors.md`. It gives
the same output when run twice. It never deletes a file, never touches an
entry with `"generated": false`, and keeps every evidence item without
`"generated": true` on a generated entry (and then the entry's status
too). Generated entries that no source produces any more stay in place,
and the build lists them as stale.

## Person A's parts export

Hanav's `person_a/mine/export_library.py` writes these files. The build
above does not change them.

- `library/catalog.jsonl`: all passing parts, one JSON record per line.
  `units` points at the part's index file in this folder, relative to the
  repo root.
- `library/<model>/<id>.pt`: index file `{kind, model, layer, indices}`. It
  lists which neurons, heads, experts, or layers make up the part. The
  weights stay on Hugging Face; fetch only these.
- `library/<model>/parts.jsonl`: the same records, for one model.

How sure each label is (`tier`):

- `screened`: the v2 light gate passed. Switching the part off dropped its
  skill by at least 15% on both the search and the hidden held-out
  questions, other skills barely moved, and the drop beat 5 random
  same-size parts.
- `confirmed`: also passed a generation check.
- `provisional`: v1 (`pipeline: v1-dissect`). These are GSM8K parts from
  the first runs. The gate changed after the first candidate, so treat
  them as provisional.

`skill.source` says whether the label came from a built-in benchmark or
from a test the agent wrote itself. Agent-written tests are frozen under
`person_a/mine/skills/`.

Updated from the mining runs with
`python person_a/mine/export_library.py --repo .`. Teammates' runs write
to `person_a/catalog/` and `person_a/parts/`, and this script copies
passing parts here. The same parts are also entries under
`components/Qwen__Qwen2.5-Math-1.5B/`, with their tier and evidence, and
each of those entries names its copy here in `exported_copy`.
