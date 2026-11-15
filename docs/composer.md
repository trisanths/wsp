# The composer

The composer is an LLM (OpenAI `gpt-6-astra`, high reasoning effort) that
writes Sapien programs for a request. It reads the atlas of models and the
catalog of mined parts, writes a program in the language of `docs/language.md`,
compiles it on Modal, reads the probe report, and revises. When it is
done, it submits one program. The benchmarks then score that program
and a set of honest rivals, once.

Three owner rules (2026-11-14) shape the whole loop:

1. Nothing that trains or fits reads a reported benchmark, in any split.
   Heal text comes from the `wsp.healdata` pools, and `healdata.check_sources`
   runs on every source that trains.
2. Nothing selects components or programs on benchmark data. Benchmarks are
   chosen per user request, so the composer does not practice on them. They
   score the final model once.
3. The practice signal is the benchmark-free probe of `wsp/probe.py`, on
   held-out pool text.

The composer only composes. It has no tool that reads or writes raw weight
values, and the language has no statement that sets one. The only training
in a compile is the stitches and the heal (glue, part LoRAs, a host LoRA,
routers, gates, and an optional distillation loss), and each stage has a
step cap.

The composer can write every Sapien statement: grafts (with `gate on`),
`experts`, `insert`, `lora`, `heal` with `distill`, `mask`, `tv`, `from`,
`carve`, `splice`, and `budget`. It cannot write `tap` or `tap_train`. A
tap runs a source model beside the host at inference, so the result is not
one network that runs alone. A program with a tap is rejected before it
compiles, with a message that says so, and the rejection costs no compile.

## Files

| File | What it holds |
|---|---|
| `composer/agent.py` | the loop, the system prompt, and `run_request` |
| `composer/tools.py` | the five tools and their checks |
| `composer/sizes.py` | size estimates, the request budget, and constraint checks |
| `composer/sources.py` | reads the atlas, `profiles.json`, and the catalogs; `sanitize` |
| `composer/skills.py` | skills, heal data names, pool ranges, benchmark set names, answer checks, the probe objective |
| `composer/probes.py` | which donor each skill is probed against, and the probe path (KL or donor text) |
| `composer/evalsets.py` | benchmark loaders and scoring, and the heal and probe text from healdata pools (runs on Modal) |
| `composer/llm.py` | the HTTPS client for the Responses API, with retries |
| `composer/archive.py` | the run archive |
| `composer/arms.py` | the comparison arms, the best existing model, and the benchmark and probe verdicts |
| `composer/comparison.py` | compiles the arms, scores every model, and builds the results |
| `composer/report.py` | the results file |
| `composer/requests/*.json` | the requests |
| `modal_composer.py` | the Modal functions and the launch entrypoint |

## Launch a request

```bash
MODAL_PROFILE=trisanth .venv/bin/modal run modal_composer.py --request composer/requests/math_plus.json
```

The loop runs on the local machine. It calls only the OpenAI API and the
Modal functions. When it stops, the entrypoint compiles the comparison
arms, scores every model on the held-out sets in parallel, writes
`results/composer_<request>.json`, and prints one verdict sentence per arm.
Add `--skip-heldout` to stop after the loop.

For a cheap check of the whole path, run
`composer/requests/smoke_qwen05.json`.

The loop runs in the local process, so keep the machine awake until it
ends. Each request can run in its own terminal, also under different Modal
profiles.

The OpenAI key is read from `.env` (`OPENAI_API_KEY=...`) at run time. It
goes only into the Authorization header of requests to `api.openai.com`.
The archive and the log pass every string through a redaction step first.

## Requests

A request is a JSON file:

```json
{
  "name": "math_plus",
  "description": "The best math model under 2B total parameters.",
  "base": null,
  "skills": {"math": 1.0},
  "constraints": {"max_total_params_b": 2.0, "max_run_params_b": 2.0, "max_heal_minutes": null,
                  "heal_steps_cap": 1000, "compile_budget": 12},
  "objective": "max mean probe transfer over math, with drift kept under the tolerance",
  "drift_tolerance": 0.05,
  "agent": {"model": "gpt-6-astra", "reasoning_effort": "high", "max_turns": 60, "max_tokens": 3000000}
}
```

`skills` maps each skill to its weight. The objective is the weighted mean
transfer over the skills minus a drift penalty (see Compile and probe). A `null` constraint has no cap.

| Field | Meaning |
|---|---|
| `base` | optional. The model id the program must use as its host. Left out, the composer picks the host. |
| `constraints.max_total_params_b` | the most parameters the compiled model holds |
| `constraints.max_run_params_b` | the most parameters one token runs |
| `constraints.max_heal_minutes` | the most minutes the heal may take |
| `constraints.heal_steps_cap` | the most steps in any training stage |
| `constraints.compile_budget` | how many compiles the loop may spend |
| `drift_tolerance` | the drift on general text, in nats, that costs nothing (default 0.05). Drift above it costs 10 objective points per nat. |
| `practice_items` | optional cap on the probe texts per skill (default 128) |
| `heldout_items` | optional cap on the items per benchmark set in the final report. A fixed random subset of the set is used. |
| `report.control_seeds` | how many random controls the report compiles (default 2) |
| `report.far_factor` | the size limit counts as far above the base at this multiple of the base (default 2.0) |
| `report.noise_floor` | the smallest benchmark objective gap that can count as a win (default 0.01) |
| `report.probe_noise_floor` | the smallest probe objective gap that can count as a win (default 0.03) |

| Request | Skills | Total params cap | Compile budget |
|---|---|---|---|
| `math_plus` | math | 2B | 12 |
| `math_code_med` | math, code, medicine | 3.5B | 15 |
| `law_small` | law | 1.5B | 12 |
| `generalist_7b` | knowledge, math, code | 8B | 15 |
| `smoke_qwen05` | math | 2B | 3 |

`smoke_qwen05` is the cheap end-to-end check. It names the base
`Qwen/Qwen2.5-0.5B-Instruct`, allows 3 compiles and 200 steps per stage,
probes on 40 texts per skill, and scores 200 items per benchmark set with one
random control. One run costs a few dollars on Modal.

## Tools

The model gets these five function tools and nothing else.

- `atlas(skill, model)`: model rows from `atlas/entries/*.json` and
  `profiles.json`. A row has the size (params, active params, layers,
  hidden), the tokenizer family, and other structure facts. Every score whose
  source is a reported benchmark is hidden, whatever the split: a score shows
  only when its name starts with `probe_`, `pool_`, `kl_`, `nll_`, or `ppl_`
  and its source note names no benchmark. The `wsp.atlas` skill scores,
  skill details, layer maps, and expert maps come from benchmark screens, so
  they are hidden too. The model sees sizes, layers, families, and
  structure.
- `catalog(type, model)`: mined parts from `catalog.jsonl` and
  `person_a/catalog.jsonl`. A row shows the part's structure (layer, units,
  width) and only the structure facts of its signature (mask sizes and
  overlaps). Measured effects are hidden: their source is a benchmark or
  unknown. Person A's parts also show a `heldout_overlap` note. Any donor
  layer range is also a graft id: `block/<org>/<model>/<first>-<last>`.
- `language_reference()`: the text of `docs/language.md` and short
  composer notes.
- `compile_and_practice(program_text, practice_tasks)`: see below.
  `practice_tasks` names the skills to probe.
- `finish(program_text, rationale)`: submit a program that compiled and
  was probed within the constraints.

`sanitize` runs on every tool result. It drops unit lists, samples, and
loss curves, and it turns any list of more than 32 numbers into a short
summary (count, min, max, mean). No tool opens a weight file.

## Compile and probe

1. `wsp.lang` parses the text. A parse error goes back to the model with
   its line number.
2. The request's limits become the program's `budget` (total params, params
   run per token as `max_active_params_b`, heal minutes). A `budget` line in
   the program can only tighten them. The compile then enforces the same
   limits on the built model before the heal.
3. The constraint check estimates the total params and the params run per
   token from the atlas sizes, including the donors' MLPs of an `experts`
   statement (the `top_k` largest MLPs per layer count as run) and the
   layers and stitches of an `insert`. It also checks that the heal and
   each stitch are at or under the step cap (a stitch without `steps`
   counts as 500, the GraftSpec default), and that the program composes
   something. It rejects:
   - a tap or `tap_train` (not Sapien);
   - a program that mixes grafts, experts, and inserts;
   - gated grafts with different `gate_bias` or `gate_lr`;
   - a distillation teacher that has another tokenizer family than the host,
     or a family the atlas does not know;
   - a host other than the request's `base`;
   - a block, insert, splice, or tap layer past the model's last layer, an
     insert `at` past the host's depth, a mask or task vector built on
     another model, a graft of a part type the compiler cannot graft, and an
     unknown heal or carve data name (`gsm8k_train`, the language default,
     is a benchmark, so every heal must write `data NAME`);
   - a combination that the compiler refuses: masks or task vectors with
     experts or inserts, an experts program with `lora ... layers`, a `lora`
     without a heal, a heal with nothing to train, and grafts, experts,
     inserts, masks, or task vectors in a program that writes one plain model
     (`from`, `carve`, `splice`);
   - a host with only a LoRA and a heal, with or without distillation, since
     that is the fine-tuning baseline;
   - a model that is not in the atlas, since it has no known size.

   A block graft counts its donor layers and its `w_in` and `w_out` glue
   (full rank, or `rank R`). The estimate leaves out the stitch network.
   The estimate counts the input embeddings as run per token and leaves out
   unmerged LoRAs, so it can sit above or below the measured numbers. The
   measured check below decides.
4. Modal compiles the program with wsp on one H200 into
   `/vol/composer/<request>/<program hash>`, and runs the probe on the
   requested skills (see Heal, probe, and benchmarks). It reports the measured total params and
   params run per token (`params_total` and `params_active` of the compile
   report, which leave out the input embedding table and the experts that
   `top_k` leaves out), the heal seconds and steps run, decode tokens per
   second at batch 1, the compile and practice seconds, a short compile
   report, and any errors.
5. The composer checks the measured numbers against the limits again. A
   program over a limit, or one whose compile did not report its size or heal
   time, cannot be submitted. A heal that stopped at the minute limit is
   within it. A compile that raises `CompileError` for the budget returns the
   measured numbers in its error.

A compile costs one unit of the compile budget. A parse error, a
constraint rejection, or a program that already ran costs nothing. When the
budget is spent, the tool refuses and asks for `finish`. When the Modal
call itself fails (a timeout, a lost container), the compile still costs
its unit, but the result is not cached, so the same program can run again.

Each result has the probe per skill (`transfer`, its donor and path, and a
verdict), `drift`, `drift_penalty`, `practice_objective` (the probed
skills), and `request_objective` (all request skills, an unprobed skill as
transfer 0). The loop ranks programs by `request_objective`, so a program
probed on one easy skill does not win the fallback.

The objective is the weighted mean transfer over the skills minus
`10 * max(0, drift - drift_tolerance)`. A skill with no donor component has
transfer 0. A skill whose donor is no better than the base on the text has no
gap, and its transfer also counts as 0.

The model may compile a program with `control random` to compare its own
components against random ones on the probe. `finish` refuses such a program, and so does
the fallback: the report builds the random control itself.

The model can grow past the base. The limits are the only bound on the size.

## Heal, probe, and benchmarks

Heal data names are `math`, `code`, `medicine`, `general`, and
`composer_mix` (an even mix of the four). Each is a `wsp.healdata` pool, so
no name reads a benchmark. `composer.skills.check_heal_sources` runs
`healdata.check_sources` on a name's dataset ids, in the pre-check
and again in `composer.evalsets.heal_texts_for` on Modal. A heal or carve
statement with any other name is rejected. The old names (`gsm8k_train`,
`math_train`, `mbpp_train`, `medmcqa_train`, `mmlu_aux`, `trivia_train`) no
longer resolve.

Each pool has one seeded order. One fetch takes the pairs up to the end of
the probe range, and two slices of it never overlap:

| Use | Pairs of the pool |
|---|---|
| heal and carve text | 2000 to 4999 |
| probe text | 5000 to 5127 |

Skills map to a heal pool and a probe pool: math to `math`, code to `code`,
medicine to `medicine`, knowledge and law to `general`. Knowledge and law have
no approved pool of their own, so their probe is on general text and says
little about those skills. That is a known weak spot.

The probe (`wsp/probe.py`) gives, for each requested skill, the transfer
against the skill's donor on the skill's held-out pool text: the share of the
base-to-donor KL gap that the compiled model closes (0 is the base, 1 is the
donor). It also gives drift: the KL from the frozen base on general pool text.

The donor of a skill is the source model of the components the program used
for it. A part tagged for another skill (by its `skill`, `domain`, `tags`,
`label`, or `trained_on` fields) does not count. A part with no skill tag
counts for every skill. With several donors, the transfer of the best one is
reported. With none, transfer is 0.

A donor with the host's tokenizer family is compared by next-token KL. Any
other donor, and a donor whose family is unknown, is compared on text the
donor generated (`probe.donor_samples`): transfer is the share of the
base's loss on that text that the compiled model removes
(`probe.transfer_by_text`, with a floor of 0). The donor's answers are cached
at `/vol/composer/donor_samples/<donor>/<skill>.json` on the Modal volume,
one file per donor and skill. That transfer is not on the same scale as the
KL transfer, so the report keeps the path next to each number.

Benchmarks are the held-out sets below. The composer never sees a benchmark
number and never selects on one. They run once, after `finish`, with fixed
settings.

| Skill | Benchmark sets, scored in the final report |
|---|---|
| math | GSM8K test, MATH-500 |
| code | HumanEval, MBPP test |
| medicine | the 1000 round 10 MedMCQA items |
| knowledge | the 1000 round 6 MMLU dev items |
| law | MMLU test, law subjects |

Multiple choice uses the log-probs of " A" to " D" after "Answer:", as in
round 6. GSM8K uses the raw "Question: ... Answer:" prompt and 256 greedy
tokens. MATH, HumanEval, and MBPP use the chat template when the tokenizer
has one, greedy decoding, and 768 or 512 tokens. Code runs in a subprocess
on the Modal container with a 10 second limit.

`heldout_items` cuts a benchmark set to a fixed random subset in its
original order. `practice_items` cuts the probe texts per skill to a prefix.

## The comparison set

After `finish`, the report scores the chosen program on the benchmarks, once,
against these rivals, all with the same heal data, the same heal settings
(learning rate, batch size, sequence length), the same seed, and the same
heal minute limit. The step counts are the program's own, except that the
host-only arms get the sum of all the program's training stages (stitches and
heal), so they are never short of steps.

| Arm | What it is |
|---|---|
| base | the host alone |
| base + LoRA | the host with a LoRA (rank 16, or the program's LoRA), healed on the same healdata text and steps as the program |
| base + distill | the same LoRA heal, on the same text and steps, with a distillation loss from one source model, for each source that has the host's tokenizer family. A source with another family is skipped, and the results say so. |
| random components | the program with `control random` on every component (random units of the same size from the same donor), for `report.control_seeds` seeds |
| best existing model | only when `max_total_params_b` is at least `report.far_factor` times the base: the best model that fits the total and run limits, picked by the atlas entries first and `profiles.json` second |
| sources | each source model of the program, as a reference |

Masks and task vectors have no random control, so a program built from them
has no random-components arm. The results list every skipped arm with its
reason. Heal minute limits can stop arms at different step counts, so the
arm compiles record `heal_steps_run`.

Every arm also gets the probe. The probe runs on the final program, each
arm, and each source with the host's tokenizer, against the final program's
donors, so the numbers compare. A model with another tokenizer than the host
(a best existing model, say) has no probe, and the row says why. The host is
the reference, with transfer 0 and drift 0 by construction.

Each arm gets a benchmark verdict and a probe verdict. The program beat an arm on the benchmarks only when its objective is
above the arm's by more than the noise threshold. The threshold is the
largest of the noise floor, twice the item sampling error of the
difference (from the held-out item counts), and the spread between the arm's
own seeds (the random controls). The verdict is `beat`, `tied`, or `lost`.
An arm that failed to compile or score has the verdict `not scored`, and so
does an arm with a held-out set that did not score, since its objective would
measure other items than the program's (the row lists `missing_sets`). Each
verdict has a plain sentence with the numbers, and the entrypoint prints
them. The probe verdict uses the probe objective, with a noise threshold of
the larger of `report.probe_noise_floor` and the spread of the arm's seeds. The
results end with a plain `summary` that says which arms the program beat by
more than noise, on the benchmark numbers and on the probe numbers, and which
it did not.

The chosen program is compiled once, so its own seed noise is not measured.
The random control spread and the item error stand in for it.

## Results

`results/composer_<request>.json` holds:

- the final program (text and JSON), the rationale, and whether the loop
  chose it (`fallback`) because the model stopped without `finish`;
- its measured size and decode speed, and its practice feedback (the probe
  report of the compile, with the objective);
- the budget used: compiles, turns, tokens, and the stop reason;
- for the final program and every arm of the comparison set, under
  `comparisons` (`base`, `lora_on_base`, `distill_on_base`,
  `random_control`, `best_existing`, `sources`): the benchmark scores per set
  and per skill with the objective, and the probe (`probe`, `probe_summary`,
  `probe_objective`: transfer per skill, drift, penalty);
- `benchmarks_scored`: the models the benchmarks ran on, each once;
- `content_gain`: per skill, the final transfer minus the mean transfer of the
  random-components arms;
- `verdicts`: one entry per arm and basis (`benchmark` or `probe`) with the
  numbers and the plain sentence, and `summary`, the plain paragraph;
- `skipped_arms`, `same_heal` (the data, steps, learning rate, and budget
  every arm used), `arm_compiles`, and `best_existing_choice` (the model, its
  earlier score, and where the score came from);
- `fits_constraints` on every scored row: whether its measured size is
  under the total params cap. The base of a carve or a big donor is often
  over the cap, so it is a reference, not a rival;
- `heldout_overlap`: the parts the final program uses that were mined on
  held-out items (today, Person A's GSM8K parts).

The earlier benchmark scores that pick the best existing model are used by
the report only, to choose a rival to score once. No tool shows them to the
model.

## Archive

Each run writes `composer/runs/<request>/<timestamp>/`: every request body,
every API reply, every tool call and tool result as numbered JSON files,
plus `log.txt` and `final.json`. The files hold the literal content, with
the key redacted.

## Limits of the loop

The loop stops at `finish`, at the turn cap, at the token budget, or on an
API error that the retries could not clear (up to 6 retries with backoff
on 408, 409, 429, and 5xx). Without `finish`, it submits the best practiced
program within the constraints.
