# Library compile test (partcheck)

Code: `wsp/partcheck.py`. Tests: `tests/test_partcheck.py`. Round script: `modal_round23.py`.

The question: does a part that a miner found in a donor model keep its skill after the compiler puts it into another model? One answer per part, with random and scrambled controls beside it. No reported benchmark is read. Training uses the miners' own skill items and `wsp.healdata` general text. Scoring uses items the mining agent never saw.

## Inputs

`load_parts` reads `library/catalog.jsonl` and every `person_a/catalog/*.jsonl` (not `*.rejected.jsonl`). A part id that appears twice for one donor is kept once, from the first file. When the copies differ (skill hash, layer, layers, unit count), `missing` gets a `duplicate id with different content` entry. An id used by two donors keeps both parts and adds an `id also used by another donor` entry. On 2026-11-14 the files hold 531 rows with 416 distinct ids; every repeat is identical. Rows of tier `screened` are taken by default. Backslashes in paths are fixed. A skill path is relative to `person_a/`. A unit path is relative to `person_a/` (per-donor files) or to the repo root (`library/`). A row that has no skill file, a missing skill file, or a missing units file is listed in `missing` and never stops the load. Rows with the skill source `builtin` or `pool` and no path read `person_a/mine/data/builtin_skills.json` or `pool_skills.json`.

A skill file holds `items` and `control`, each `{"search": [...], "heldout": [...]}`. An item is `{context, continuation, digits_only}`. The row hash is compared with the file hash. A mismatch is kept in `hash_ok` and in the report (`hash_mismatch`). It does not stop the run. On 2026-11-14 the K2-Horizon-7B rows show three mismatches.

Split. The seam trains on `search`. Every score is measured on `heldout`. There is no hash split of our own.

The miners' two halves are not independent. On 2026-11-14, across 110 distinct skill files, 1355 of 5354 heldout items repeat a search item (59 files; `boolean_logic`, `month_successor`, `json_literals`, and `compass_opposite` are fully duplicated), and 1453 heldout control items repeat a search or control item. `check_parts` drops every heldout item and heldout control item whose context equals a search context, whose (context, continuation) pair is in the search half, or whose context and continuation text sits inside a search context (a few-shot demo). `min_items` applies to what remains. The row keeps `n_heldout_dropped_overlap`. A skill with nothing left is skipped.

When `--max-parts` cuts the list, parts with 64 units or more (and blocks) come first, then one part per skill by round robin. Single neurons have never beaten random in a compile.

## Measures

For a set of items `i = 1..n` with prompt `c_i`, answer `a_i`, and flag `d_i` (digits only):

    real(M)     = sum_i sum_t log p_M(a_i[t] | c_i, a_i[<t])  /  (number of scored tokens)
    shuffled(M) = the same with a_{(i+1) mod n} given c_i
    margin(M)   = real(M) - shuffled(M)

Tokens are pooled over all items. A token is scored when `d_i` is false, or when its text holds a digit. This is the miners' `margin_vs_shuffled` (`person_a/mine/engine_app.py`, methods `base` and `_score_rows`). The prompt is encoded with special tokens, the answer without, and the ids are cut at 1024. Each model uses its own tokenizer.

Per part `p`, on the held-out items (B = base, D = donor, C = compiled, C_off(p) = compiled with only part p's gate closed):

    gap          = margin(D) - margin(B)
    gain         = margin(C) - margin(B)                      (mean over seeds)
    recovered    = gain / gap                                  (only when gap >= min_gap)
    contribution = margin(C) - margin(C_off(p))                (mean over seeds)
    control_best = max( mean contribution of the random compiles, mean contribution of the scrambled compiles )
    content_z    = min over control kinds of (contribution - mean of control contributions) / std of control contributions
                   (probe.content_z; needs two or more seeds that differ)
    control_effect = margin on the held-out matched control items, compiled minus base
    drift        = KL(base || compiled with only part p open) on held-out general text
    drift_ref    = median of the same drift over the control compiles of that part

Gain, recovered, and the margins are properties of the whole compile: every part of a group shares one set of translation layers, so two parts of one group get the same gain. Only the contribution tests one part. Each control compile is therefore measured the same way: its margin minus its margin with that part's gate closed, per part, per seed. The real part must beat those control contributions. `control_best_gain` (the group-level gain of the best control) is kept in the report for reading only. Parts of one skill cover for each other when one gate is closed, so a contribution near 0 for a part with siblings is expected, and the control contributions have the same effect.

Random control: same compile with random same-size parts of the same donor layer. Scrambled control: same layers and units, weights rewritten by `wsp.scramble`. Both use every control seed.

Margins are noisy at small n. A heldout half of 24 to 60 items has a standard error that can reach 0.2 nats.

## Verdict rules

Applied in this order, thresholds in `partcheck.THRESHOLDS`:

1. `harmful`: contribution below minus 0.05.
2. `no gap`: gap below 0.1 (the base is already as good as the donor, or the donor lacks the skill).
3. `harmful`: drift above `max_drift` (0.1) plus `drift_ref`. Drift is measured with the part open on chat-formatted text and can reach 2 to 6 nats for any compile, so the reference is the control compiles' own drift. With no control run `drift_ref` is 0 and the 0.1 limit alone applies: read such a row with care, and read the drift numbers of the smoke before trusting any `harmful`.
4. `no better than controls`: contribution at most 0.05, or contribution less than the best control contribution plus 0.1, or content_z below 2 when it exists.
5. `survives provisional`: every rule above passed with fewer than 5 seeds (`min_seeds_survive`). With three control seeds a z of 2 is weak. `confirm_seeds` (modal default 3,4) reruns the real and control compiles for those seeds when a part is provisional, and the verdict is recomputed with all seeds.
6. `survives compile`: otherwise, with at least 5 seeds.

Every row prints `n_seeds` (the smaller of the real and control seed counts).

Without a control run, rule 4 rests on the contribution alone. Say so when reading such a row.

Group verdict, from the group compile of all parts together: `no part survives` or `parts survive together`. With `per_part`, each surviving part is compiled alone (seed 0, no controls). `alone_share = (margin alone - margin base) / gain in the group`. A part with `alone_share` under 0.5 needs the others, and the group verdict becomes `some parts need the others`, or `parts survive together and alone`.

## Output

One JSON per donor (`results/round23_<donor key>.json`, a list of reports, one per base) with the full report. One rows file, `results/round23_rows.json`, merged by (id, base). The merge key is (id, base): ids are unique per donor, and a repeated id from another donor is listed in `missing`. Each row has:

    id, base, relation, model, margin_base, margin_donor, margin_compiled, recovered,
    contribution, content_z, control_effect, drift, verdict, reason, n_search,
    n_heldout, n_heldout_dropped_overlap, n_seeds, skill_hash_ok

A part that was not compiled still gets a row, with `verdict: "skipped"` and a `reason` (too few items after the overlap drop, missing skill or units file, refused model type). A duplicate-id entry gets no row. On Modal each cell writes its partial report to `/vol/results/round23/` after every compile (`"partial": true`), so a time limit or a crash keeps what finished.

followed by the miner fields the part had: `signature_z`, `signature_guard`, `selection_data`, `routing`, `trained_on`, `evidence_used`, `created`.

Merge into library rows by (`id`, `base`): set a new field (for example `compile_check`) on the library row with the same id to the row's values, keyed by `base`. A part can have one row per base. Rows are never matched by position.

## Run limits (modal_round23.py)

- At most `--max-containers` cells run at once (default 4). Models that two cells share are downloaded once first (`warm_models`).
- More than 3 cells are refused until `results/round23_rows.json` has a row, unless `--force`. A tf5 run is refused until `--tf5 --smoke` has printed a row.
- A donor with more than 24 parts is cut to 24, large parts first (`--max-parts` overrides). Stage 1 gets 100 steps per part: the step count is raised to `100 * parts / components_per_step` (63 parts: 2100 steps, not 300). The old 300 steps gave 14 steps per part and the controls would have won.
- No compile starts after `--wall-hours` (default 5.2); the Modal timeout is 6 hours.
- OLMo-2 cells are dropped: `wsp.parts.find_layer_modules` takes `post_attention_layernorm` as the pre-MLP norm, but OLMo-2 applies it to the attention output and feeds the raw residual to the MLP. `partcheck.UNSUPPORTED_MODEL_TYPES` refuses the model type. Granite (muP multipliers) and K2-Horizon (custom code) have not been checked by any smoke.
- The Qwen3.8-27B to Qwen3.5-9B pair and the gemma-4-31B to gemma-4-E4B pair have equal vocabularies (248044 and 262144 tokens, compared from the Hub `tokenizer.json` files; the files differ), so logit KD applies.

## Unverified

- No Modal run exists. The CPU tests use tiny random models, and one CPU smoke used real catalog rows (Qwen2.5-Math-1.5B, three parts) with tiny random Qwen2 models and the real Qwen tokenizer. That checked the file formats and both part kinds, not the science.
PLANTED_PLACEHOLDER
- Thresholds (`min_gain` 0.1 nats, `min_z` 2, `max_drift` 0.1, `min_seeds_survive` 5) are guesses, not calibrated on a real run.
- The drift of a part is measured with the part open on every token. The token gate normally closes on general text. The control median is the reference, but no real run has shown the distribution yet.
- The base for the four 5.x donors is chosen from tokenizer file hashes only and lineage was not run.
- Qwen2.5-Coder-7B into Qwen2.5-1.5B-Instruct has vocabulary sizes 152064 and 151936. Whether `wsp.synth` then uses logit KD or LM loss is not checked.
- Most Qwen2.5-Math-1.5B parts hold 1 to 4 units. Expect `no better than controls` there.
- `skill_hash` mismatches are recorded and the part is still run.
