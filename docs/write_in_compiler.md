# Write-in through the compiler front door (2026-11-15)

The cross-family write-in of `docs/write_in_xfam.md` is now a compiler method. A request file goes in, a saved and
scored network comes out. This page says what was built, what one compile measured, and what was not run.

## One command

    modal run --detach modal_sapien.py --request demo/requests/xfam_knowledge.json

The request says: give Llama-3.2-1B-Instruct the general knowledge of Qwen2.5-7B-Instruct (seed 1). The saved plan is
`demo/plans/xfam_knowledge.txt`. `sapien plan demo/requests/xfam_knowledge.json` prints it without a GPU. Add `--smoke`
for a 3-step check on the real models (it writes `/vol/sapien/xfam_knowledge_smoke`).

## How the planner chooses it

`wsp/lineage.py` calls the pair "foreign" (another family or tokenizer). For a foreign source, a dense donor, and a
skill whose pool is knowledge, medicine, or law (four-option style), `wsp/auto.py` lists `pgraft` first and `synth`
behind it. The first method that fits the limits wins, so a request with limits too small for the donor slice falls
back to `synth` and the plan says which limit blocked `pgraft`. Same-lineage sources (fine-tunes, other sizes of one
family) and other skills keep their old methods. A request can force the method with `"method": "pgraft"`; a source
that does not qualify is refused. Two sources, or a weave or slim stage in the same request, are refused for `pgraft`.

The plan prints: host layer, donor span, Enc and Dec sizes, the data pool, steps, params per token after the write-in,
and the controls. Sizes come from configs alone. For the demo request the plan says 5,213,011,330 parameters per token,
and the stored row of the research driver has exactly 5,213,011,330.

Layer rule, scaled from the one measured pair: host layer is host layers // 2 - 1 (7 of 16), donor span is donor
layers // 7 to 5 * donor layers // 7 (4..20 of 28). Other pairs use the same rule without a measurement behind it.

## What the compiler runs

`wsp/auto.py make()` sends a one-stage `pgraft` plan to `wsp/pgraft_stage.py run_make`. The text probes of the other
methods are skipped, because they need a shared tokenizer. Steps:

1. Load the host and donor, read the cached pool and eval sets of the wi_xfam rows (`/vol/wi_xfam/v1`, built from
   scratch when missing), and score the host.
2. `wsp/xfam.py train_network`: functional alignment of Enc to the donor (600 steps), then 4000 task steps with the
   answer-letter loss, the donor soft-letter KL, the drift KL to the frozen host, and the alignment regularizer.
3. Save `<out>/pgraft_v1/` (`meta.json`, `glue.safetensors`, `lora.safetensors`). It holds Enc, Dec, the gate, the
   LoRA, the donor sink vector, the spec, a fingerprint of the host, and the ids of the host and donor. About 106 MB.
   The donor slice is read from the donor by id at load time; the host is unchanged. The format version is in the path
   and in `meta.json`; the loader refuses another version or a host whose fingerprint differs.
4. Load the artifact into a fresh host (`xfam.load_artifact`) and score that network. The report scores come from the
   reloaded network. `python -m wsp.xfam score <dir>` does the same from the command line.
5. Train the identity and scrambled-donor controls with the same code and data (setting `pgraft_controls`), writing
   the report after each arm.

`modal_wi_xfam.py` now imports its data, scoring, and decode helpers from `wsp/xfam.py`, so the research driver and
the compiler score with the same functions (a test checks `score_letters` against `modal_round6.score_mmlu`). The
training loop of the graft arm exists twice: in `modal_wi_xfam.py _cell` (all arms) and in `xfam.train_network`
(graft, identity, scrambled). They carry the same recipe, and the compile below is the check that they agree.

## Result of one compile (seed 1, one seed)

Held-out items nothing trained on. Trained on the cached pool (MMLU auxiliary_train, CommonsenseQA, OpenBookQA, QASC,
LogiQA train; eval prompts removed). Full table with sources: `results/wi_compiler/table.md`.

| arm | MMLU held-out 2000 | ARC-C test | OBQA test | params/token | GPU h |
|---|---|---|---|---|---|
| base | 44.6 | 55.3 | 60.0 | 1.236B | |
| graft, compiler path (seed 1) | 58.7 | 84.5 | 88.6 | 5.213B | 0.68 |
| graft, wi_xfam driver (seed 1) | 59.5 | 85.2 | 87.0 | 5.213B | 0.66 |
| identity control, compiler path | 50.4 | 70.7 | 76.6 | 1.251B | 0.31 |
| identity control, wi_xfam driver | 50.9 | 69.9 | 77.4 | 1.251B | 0.32 |
| scrambled-donor, wi_xfam driver | 50.3 | 70.1 | 78.6 | 5.213B | 0.64 |

The compiler graft is 0.8 MMLU points under the wi_xfam graft of the same seed, inside the 1.5 point bar. The base
scores are identical to the digit (44.6, 55.3, 60.0). The alignment report is identical to the wi_xfam row (KL 0.5152,
survival 0.907) and the training token count is the same (10,317,916), so the data order and stage 1 reproduce exactly.
The final scores differ because the two code paths draw random numbers in a slightly different order after that
point (the LoRA and glue initialisation), and seed spread in these rows is about 1 point. Reloading the artifact
changed no score (gap 0.0 on MMLU held-out). With the graft switched off the saved network scores 45.5 held-out.

Decode speed of the reloaded network is 30.2 tokens per second at batch 1 (hook-based path, one measurement) against
94.6 for the host in the same container.

## What was not run

- One seed. A claim needs three seeds for the graft and the strongest control; the wi_xfam rows have them
  (59.6 +-0.8 over seeds 1, 2, 3) and this compile adds one more number at seed 1.
- The scrambled-donor control of the compile. The `trisanth` Modal workspace was disabled while that control was at
  step 4300 of 4600 ("workspace is disabled"), so the row was lost. The scrambled row in the table is the wi_xfam
  driver's seed 1 row. The report on the volume holds the graft and identity arms.
- Matched LoRA, rank-128 LoRA, and full fine-tune are not run by the compiler. The plan names them and the table
  uses the wi_xfam rows of seed 1 (50.5, 50.6, 51.1).
- Other host and donor pairs, other seeds, and the Llama-3.2-3B host. An earlier 1-seed run on that host only tied LoRA.
- Decode speed beyond the hook-based path; no fused kernels.

## Cost

Smoke 0.1 GPU hours, the compile about 2 container hours on one H200 (graft 0.68, identity 0.31, scrambled lost near the
end). Under 15 dollars at list price.
