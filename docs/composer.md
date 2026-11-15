# The composer

The composer is an LLM (OpenAI `gpt-6-astra`, high reasoning effort) that
writes Sapien programs for a request. It reads the atlas of models and the
catalog of mined parts, writes a program in the language of `docs/language.md`,
compiles it on Modal, reads the practice scores, and revises. When it is
done, it submits one program. The held-out tests then score that program
and a set of honest rivals, once.

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
| `composer/skills.py` | skills, practice and held-out sets, answer checks, the objective |
| `composer/evalsets.py` | data loaders, scoring, and the composer heal data (runs on Modal) |
| `composer/llm.py` | the HTTPS client for the Responses API, with retries |
| `composer/archive.py` | the run archive |
| `composer/arms.py` | the comparison arms, the best existing model, and the verdicts |
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
  "description": "The best math model (GSM8K and MATH) under 2B total parameters.",
  "base": null,
  "skills": {"math": 1.0},
  "constraints": {"max_total_params_b": 2.0, "max_run_params_b": 2.0, "max_heal_minutes": null,
                  "heal_steps_cap": 1000, "compile_budget": 12},
  "objective": "max math score",
  "agent": {"model": "gpt-6-astra", "reasoning_effort": "high", "max_turns": 60, "max_tokens": 3000000}
}
```

`skills` maps each skill to its weight. The objective is the weighted mean
of the skill scores. A `null` constraint has no cap.

| Field | Meaning |
|---|---|
| `base` | optional. The model id the program must use as its host. Left out, the composer picks the host. |
| `constraints.max_total_params_b` | the most parameters the compiled model holds |
| `constraints.max_run_params_b` | the most parameters one token runs |
| `constraints.max_heal_minutes` | the most minutes the heal may take |
| `constraints.heal_steps_cap` | the most steps in any training stage |
| `constraints.compile_budget` | how many compiles the loop may spend |
| `practice_items`, `heldout_items` | optional caps on the items per practice set and per held-out set. A fixed random subset of the set is used. |
| `report.control_seeds` | how many random controls the report compiles (default 2) |
| `report.far_factor` | the size limit counts as far above the base at this multiple of the base (default 2.0) |
| `report.noise_floor` | the smallest objective gap that can count as a win (default 0.01) |

| Request | Skills | Total params cap | Compile budget |
|---|---|---|---|
| `math_plus` | math | 2B | 12 |
| `math_code_med` | math, code, medicine | 3.5B | 15 |
| `law_small` | law | 1.5B | 12 |
| `generalist_7b` | knowledge, math, code | 8B | 15 |
| `smoke_qwen05` | math | 2B | 3 |

`smoke_qwen05` is the cheap end-to-end check. It names the base
`Qwen/Qwen2.5-0.5B-Instruct`, allows 3 compiles and 200 steps per stage,
practices on 40 items per set, and scores 200 items per held-out set with one
random control. One run costs a few dollars on Modal.

## Tools

The model gets these five function tools and nothing else.

- `atlas(skill, model)`: model rows from `atlas/entries/*.json` and
  `profiles.json`. A row has the size (params, active params, layers,
  hidden), the tokenizer family, and skill scores. Atlas rows also carry
  layer maps and expert maps when the atlas has them. Scores measured on
  test items that overlap this run's held-out sets are left out:
  `mmlu_dev`, its alias `mmlu`, `mmlu_heldout` (round 7, MMLU test),
  `medmcqa`, any profile score whose note says it used test items, and the
  same splits in the law subject scores. `wsp.atlas` screens knowledge,
  medicine, law, math, and code on this run's held-out sets, so the atlas
  skill scores and skill details for those five skills are left out too.
  It builds the math layer and expert maps from MATH-500 test problems, so
  the math maps are left out.
- `catalog(type, model)`: mined parts from `catalog.jsonl` and
  `person_a/catalog.jsonl`, with short signatures. Person A's parts were
  mined and scored on GSM8K test items, so their rows show a
  `heldout_overlap` note in place of the signature. Any donor layer range
  is also a graft id: `block/<org>/<model>/<first>-<last>`.
- `language_reference()`: the text of `docs/language.md` and short
  composer notes.
- `compile_and_practice(program_text, practice_tasks)`: see below.
- `finish(program_text, rationale)`: submit a program that compiled and
  practiced within the constraints.

`sanitize` runs on every tool result. It drops unit lists, samples, and
loss curves, and it turns any list of more than 32 numbers into a short
summary (count, min, max, mean). No tool opens a weight file.

## Compile and practice

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
     unknown heal or carve data name;
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
   `/vol/composer/<request>/<program hash>`, and scores it on the practice
   sets of the requested skills. It reports the measured total params and
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

Each result has `practice_objective` (the practiced skills) and
`request_objective` (all request skills, an unpracticed skill as 0). The
loop ranks programs by `request_objective`, so a program practiced on one
easy skill does not win the fallback.

The model may compile a program with `control random` to test its own
components on practice items. `finish` refuses such a program, and so does
the fallback: the report builds the random control itself.

The model can grow past the base. The limits are the only bound on the size.

## Practice and held-out sets

Practice sets come from train or validation splits, 100 to 200 items per
skill. Held-out sets are test splits. They run once, after `finish`, with
fixed settings.

| Skill | Practice | Held-out |
|---|---|---|
| math | 100 GSM8K train (search split), 100 MATH train | GSM8K test, MATH-500 |
| code | 150 MBPP train | HumanEval, MBPP test |
| medicine | 200 MedMCQA train | the 1000 round 10 MedMCQA items |
| knowledge | 200 MMLU validation | the 1000 round 6 MMLU dev items |
| law | up to 200 MMLU validation and dev items from the law subjects | MMLU test, law subjects |

Multiple choice uses the log-probs of " A" to " D" after "Answer:", as in
round 6. GSM8K uses the raw "Question: ... Answer:" prompt and 256 greedy
tokens. MATH, HumanEval, and MBPP use the chat template when the tokenizer
has one, greedy decoding, and 768 or 512 tokens. Code runs in a subprocess
on the Modal container with a 10 second limit.

The heal data names are `gsm8k_train`, `mmlu_aux`, `trivia_train`, and
four that the composer adds: `math_train`, `mbpp_train`, `medmcqa_train`,
and `composer_mix` (an even mix). Each uses the train items after the
practice items, so heal and practice never share items. Heal data and
carve domain data always use seed 0, whatever the program seed: the seed
picks the GSM8K split, and any other seed would move practice items into
`gsm8k_train`. The program seed still sets the heal's batch order.

`practice_items` and `heldout_items` in a request cut a set to a fixed random
subset in its original order, so a cheap run uses fewer items of the same
sets. Practice items, heal data, and held-out items stay apart under a cap:
the cap only drops items. The atlas scores measured on held-out items stay
hidden from the model, as the Tools section says.

## The comparison set

After `finish`, the report scores the chosen program on the held-out sets
against these rivals, all with the same heal data, the same heal settings
(learning rate, batch size, sequence length), the same seed, and the same
heal minute limit. The step counts are the program's own, except that the
host-only arms get the sum of all the program's training stages (stitches and
heal), so they are never short of steps.

| Arm | What it is |
|---|---|
| base | the host alone |
| base + LoRA | the host with a LoRA (rank 16, or the program's LoRA), trained on the program's heal data |
| base + distill | the same LoRA heal with a distillation loss from one source model, for each source that has the host's tokenizer family. A source with another family is skipped, and the results say so. |
| random components | the program with `control random` on every component (random units of the same size from the same donor), for `report.control_seeds` seeds |
| best existing model | only when `max_total_params_b` is at least `report.far_factor` times the base: the best model that fits the total and run limits, picked by the atlas entries first and `profiles.json` second |
| sources | each source model of the program, as a reference |

Masks and task vectors have no random control, so a program built from them
has no random-components arm. The results list every skipped arm with its
reason. Heal minute limits can stop arms at different step counts, so the
arm compiles record `heal_steps_run`.

Each arm gets a verdict. The program beat an arm only when its objective is
above the arm's by more than the noise threshold. The threshold is the
largest of the noise floor, twice the item sampling error of the
difference (from the held-out item counts), and the spread between the arm's
own seeds (the random controls). The verdict is `beat`, `tied`, or `lost`.
An arm that failed to compile or score has the verdict `not scored`, and so
does an arm with a held-out set that did not score, since its objective would
measure other items than the program's (the row lists `missing_sets`). Each
verdict has a plain sentence with the numbers, and the entrypoint prints
them.

The chosen program is compiled once, so its own seed noise is not measured.
The random control spread and the item error stand in for it.

## Results

`results/composer_<request>.json` holds:

- the final program (text and JSON), the rationale, and whether the loop
  chose it (`fallback`) because the model stopped without `finish`;
- its measured size and decode speed, and its practice scores;
- the budget used: compiles, turns, tokens, and the stop reason;
- held-out scores per set and per skill, and the objective, for the final
  program and every arm of the comparison set, under `comparisons`
  (`base`, `lora_on_base`, `distill_on_base`, `random_control`,
  `best_existing`, `sources`);
- `verdicts`: one entry per arm with the numbers and the plain sentence;
- `skipped_arms`, `same_heal` (the data, steps, learning rate, and budget
  every arm used), `arm_compiles`, and `best_existing_choice` (the model, its
  earlier score, and where the score came from);
- `fits_constraints` on every scored row: whether its measured size is
  under the total params cap. The base of a carve or a big donor is often
  over the cap, so it is a reference, not a rival;
- `heldout_overlap`: the parts the final program uses that were mined on
  held-out items (today, Person A's GSM8K parts).

The earlier scores that pick the best existing model are used by the
report only. No tool shows them to the model.

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
