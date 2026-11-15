# Parts library

Every validated part mined so far, ready for the compiler. One folder per donor model.

- `library/catalog.jsonl`: all parts, one JSON record per line. `units` points at the part's index
  file in this folder, relative to the repo root.
- `library/<model>/<id>.pt`: index file `{kind, model, layer, indices}`. It lists which neurons,
  heads, experts or layers make up the part. The weights stay on Hugging Face; fetch only these.
- `library/<model>/parts.jsonl`: the same records, for one model.

How sure each label is (`tier`):
- `screened`: the v2 light gate passed. Switching the part off dropped its skill by at least 15%
  on both the search and the hidden held-out questions, other skills barely moved, and the drop
  beat 5 random same-size parts.
- `confirmed`: also passed a generation check.
- `provisional`: v1 (`pipeline: v1-dissect`). These are GSM8K parts from the first runs. The gate
  changed after the first candidate, so treat them as provisional.

`skill.source` says whether the label came from a built-in benchmark or from a test the agent wrote
itself. Agent-written tests are frozen under `person_a/mine/skills/`.

Updated from the mining runs with `python person_a/mine/export_library.py --repo .`. Teammates'
runs write to `person_a/catalog/` and `person_a/parts/`, and this script copies passing parts here.
