# Review of the library compile test (round 23), not yet fixed

The review below found real defects. modal_round23.py refuses to run until they are fixed. Written 2026-11-14 by a review agent; the fix agent was stopped by a rate limit.

Review of the partcheck build. I read the files and ran the tests (43 passed). I also queried the Hub API and ran the planted fixture by hand. I edited nothing.

1. **wsp/partcheck.py:845-850 (check_parts).** The "heldout" half is not independent of the "search" half in the real skill files.
   - Failing case: across the 104 distinct skill files now loaded, 59 have heldout items whose (context, continuation) pair is also in the search half. That is 1355 of 5127 heldout items. `boolean_logic.json` has 60 of 60 duplicated, `month_successor.json` 60 of 60, `json_literals.json` 45 of 45, `compass_opposite.json` 40 of 40.
   - Effect: the seam trains on these exact items and then scores them. Memorized answers count as retained skill, which inflates gain, contribution, and recovered.
   - Fix: after reading both halves, drop every heldout item and control item whose context or (context, continuation) is in the search half. Apply `min_items` to what remains. Record `n_heldout_dropped_overlap` in the row.
   - Not checked: whether heldout questions also appear inside the few-shot demos of search contexts. Test that too.

2. **wsp/partcheck.py:881-917 (per-part gain and control comparison).** gain, content_z, and control_best are group-level numbers copied to every part, so they do not test one part.
   - Evidence: in the planted fixture with controls and 3 seeds, `mem` and `decoy` both got margin_compiled 3.5366 and gain 4.871, with the same z (0.35). The decoy block is an identity in the donor and has contribution 0.
   - Real-data case: Math-1.5B has 18 parts for 3 skills. Parts sharing a skill get the same gain, and each other's siblings cover for a closed gate, so contribution is also near 0.
   - Fix: take the per-part control value as the control compile's margin minus the control compile's margin with that part closed. Compare `contribution` to the control contributions for the z score and the min_gain test. Or run controls that replace one part at a time. Rule: "survives" needs a real contribution that beats the control contributions.

3. **tests/test_partcheck.py:454-470 and 484 (planted test).** The test would still pass if the part did not matter.
   - The survive test runs with `controls=()`, one seed, and `max_drift` raised to 100. Per the author's report and the docs, a random block with a trained translation reaches about the real gain.
   - My 3-seed run with controls gave random margins of -1.33, 4.11, and 4.42, and scrambled margins of 1.46, 1.84, and 3.04. The real compile scored 3.54 and got "no better than controls". Two random seeds beat the real part, so the test world cannot separate content from glue.
   - The controls test at line 484 only asserts what the verdict code already does, so it can never fail.
   - Fix: build a world the glue cannot learn, such as many keys, a tiny search half, and a wide donor memorizing a hash-like map. Then assert on the heldout half that real contribution beats the random control contribution by `min_gain` over at least 3 seeds. Add a negative case with the real part swapped for `control="random"` that must not get "survives compile". Drop the `max_drift` override by calibrating the drift test instead.

4. **wsp/partcheck.py:238-253 (load_parts dedupe).** Dedupe by part id across all files.
   - Evidence: the files hold 471 rows and 356 distinct ids, so 115 ids repeat. All 115 repeats are identical today (same model, layer, units, and skill hash).
   - The brief says ids are unique per donor file, so a collision between two donors would silently drop a part.
   - Fix: key `seen` by `(model, id)` and report a `duplicate id with different content` entry in `missing` when the model or hash differs.

5. **modal_round23.py:237-245 and 274 (timeout and no partial results).** Cells can exceed the 6-hour limit and then save nothing.
   - Qwen2.5-Math-7B-Instruct has 63 parts, and 3 seeds with real, random, and scrambled compiles is 9 compiles. Each compile runs translate_in and translate_out per part, with 100 `steps_in` each, which I estimate at 1 to 2 minutes per part. That is 1 to 2 hours per compile, so roughly 9 to 18 hours.
   - Nanbeige 33B has 14 parts, and its KD uses a 33B teacher forward at every component step, so it is near the limit.
   - `save_report` runs only at the end, so a timeout loses everything.
   - Fix:
     - Write the partial report after every compile, and per-part margins after each control seed.
     - Default `--max-parts 24` for any donor with more than 24 parts.
     - Keep the 64-plus-unit preference.
     - Set `components_per_step` so each part gets at least 100 steps. Currently 300 steps × 3 / 63 parts is about 14 steps per part, so the compile is undertrained and controls will win.
   - Also add a per-cell wall limit.

6. **wsp/partcheck.py:596-599 (heal budget per part).** The step budget does not scale with the number of parts. 18 Math-1.5B parts get 50 steps each, and 14 to 18 parts per donor get 50 to 60. Raise `stage1_steps` to about `100 * n_parts / 3`, or cap the part count as in item 5.

7. **wsp/partcheck.py:881 and 891-892 (log lines).** LESSONS asks for the cell name on every score line, and the compiled and control lines carry no cell name.
   - `qwencoder7b` and `qwencoder7b_small` share a donor id (`Qwen/Qwen2.5-Coder-7B-Instruct`) and run at the same time, so their lines cannot be told apart.
   - Fix: pass a `cell` string (key plus base id) into `check_parts` and print it on every `partcheck ...` line.

8. **modal_round23.py:270-275 (concurrency).** `main` spawns every cell at once.
   - A full run with no `--only-donors` starts about 9 H200 containers, which LESSONS says is near the workspace cap of about 10.
   - Several cells also download the same weights into `/vol/hf` at the same time. For example Qwen2.5-7B-Instruct is a base for both `qwenmath7b` and `qwencoder7b`.
   - Fix: add a `--max-containers` option (default 4), and warm shared bases first (`snapshot_download` plus a commit in a prep function).
   - Fix: make the smoke gate real. Have `main` refuse a run of more than 3 cells unless `results/round23_rows.json` has at least one row, or `--force` is passed.
   - The tf5 image has its own untested path, so smoke it once with `--tf5 --smoke` before the tf5 cells.

9. **modal_round23.py:50-51 (docstring).** The note says Qwen3.8-27B → Qwen3.5-9B and gemma-4-31B → gemma-4-E4B have different tokenizers. They do not.
   - My check of the Hub `tokenizer.json` vocab hash shows Qwen3.6, 3.8, 3.5-4B, and 3.5-9B are all `70f9ced1` (vocab 248320). Both gemma-4 models are `36d412e2` (vocab 262144).
   - So KD applies there. Update the notes and the lineage claim.

10. **wsp/parts.py:139-150 (OLMo-2 cells, existing code that partcheck inherits).** `find_layer_modules` assumes `post_attention_layernorm` is the pre-MLP norm.
    - OLMo-2 (transformers 4.57.1 `Olmo2DecoderLayer.forward`) applies `post_attention_layernorm` to the attention output. Its MLP takes the raw residual and then goes through `post_feedforward_layernorm`.
    - All 9 OLMo-2-7B rows are `mlp_cluster`, so every part is built with the wrong input norm and the translate_in capture is wrong. Results would be invalid with no error.
    - Fix: hold the `olmo2_7b` cell until `find_layer_modules` handles this, or have `run_donor` skip `model_type == "olmo2"`. Run a tiny Olmo2 compile test first. Granite (muP multipliers) and K2-Horizon (custom code) are also unchecked in the smoke.

11. **wsp/partcheck.py:773-776 and THRESHOLDS (drift rule).** `harmful` is the first rule and fires when drift is above 0.1 nats.
    - Drift is measured with the part open on chat-formatted ultrachat text, and the author's tiny run saw 2 to 6 nats. The planted test only passes by setting `max_drift=100`.
    - On a real compile the first `harmful` rows may be set by the threshold and not by damage.
    - Fix: smoke first and read the drift distribution. Use the closed-gate compile and the random-control drift as the reference, and call a part harmful only when its drift exceeds the control median by a margin. Move drift after "no gap".

12. **wsp/partcheck.py:889-899 and 915-920 (z with few seeds).** Verdict at seeds 0,1,2: `content_z` uses 3 control values, and a `std` of 3 values makes z of 2 weak. The smoke has 1 seed and no z at all, so "survives" there rests only on gain beating the control mean. Print `n_seeds` per row. Use at least 5 control seeds for any row whose verdict is "survives compile", and rerun survivors with extra seeds.

13. **wsp/partcheck.py:1020-1040 (rows output, merge by id).** Rows are keyed `(id, base)`, but the docs say merge by `id` alone.
    - `rows_from_report` also leaves out parts that were skipped for too few items or that have missing files. A miner then sees no row and cannot tell why.
    - Fix: emit rows for skipped parts with `verdict: "skipped"` and a `reason`. Add `model`, `n_search`, `n_heldout`, `n_seeds`, and `skill_hash_ok`. State in the docs that the merge key is `(id, base)`.

Checks that passed:
- (c) Arguments to `synthesize`, `probe.drift`, `probe.content_z`, `lineage.relation(base, source)`, `healdata.pool`, and `healdata.check_sources` all exist and are called correctly. Coder-7B into 1.5B works: `same_vocab` compares tokenizer vocab dicts, and `kl_tokens` truncates to the shared vocabulary.
- (b) The margin matches `engine_app.base`: next-item shuffled answer, digit mask from the item's own prompt, 1024 cut, pooled tokens.
- (d) `forced_off` makes `Branch.forward` return `h`, so closing one gate isolates that branch, apart from the sibling-part effect in item 2.
- (a) The image pip list matches modal_round20 plus tiktoken for tf5. There are no hard links, and reload is wrapped.
- (g) All 21 model ids exist, none are gated, and all vocab hashes match the stated tokenizers. Nanbeige 166912 vs 166144 and Qwen 152064 vs 151936 differ only in padded embedding size.
- (h) A 33B donor at about 66 GB, a 3B base, and one copy fit on one H200. Time and memory are the open question in items 5 and 6.

Files reviewed (all under `/Users/pyro/Research/wsp-deletion-masks`):
- `wsp/partcheck.py`
- `tests/test_partcheck.py`
- `modal_round23.py`
- `docs/partcheck.md`