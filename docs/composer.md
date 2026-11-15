# The composer

The composer is an LLM (OpenAI `gpt-6-astra`, high reasoning effort) that
writes wsp programs for a request. It reads the atlas of models and the
catalog of mined parts, writes a program in the wsp language, compiles it
on Modal, reads the practice scores, and revises. When it is done, it
submits one program, and the held-out tests run once on that program.

The composer only composes. It has no tool that reads or writes raw weight
values, and the wsp language has no statement that sets one. The only
training in a compile is the stitch, the heal (glue, part LoRAs, and a host
LoRA), and tap training for taps, and each of those stages has a step cap.

## Files

| File | What it holds |
|---|---|
| `composer/agent.py` | the loop, the system prompt, and `run_request` |
| `composer/tools.py` | the five tools and their checks |
| `composer/sizes.py` | size estimates and constraint checks |
| `composer/sources.py` | reads the atlas, `profiles.json`, and the catalogs; `sanitize` |
| `composer/skills.py` | skills, practice and held-out sets, answer checks, the objective |
| `composer/evalsets.py` | data loaders, scoring, and the composer heal data (runs on Modal) |
| `composer/llm.py` | the HTTPS client for the Responses API, with retries |
| `composer/archive.py` | the run archive |
| `composer/report.py` | the comparison list and the results file |
| `composer/requests/*.json` | the requests |
| `modal_composer.py` | the Modal functions and the launch entrypoint |

## Launch a request

```bash
MODAL_PROFILE=trisanth .venv/bin/modal run modal_composer.py --request composer/requests/math_plus.json
```

The loop runs on the local machine. It calls only the OpenAI API and the
Modal functions. When it stops, the entrypoint compiles the LoRA
comparison, scores every model on the held-out sets in parallel, and writes
`results/composer_<request>.json`. Add `--skip-heldout` to stop after the
loop.

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
  "skills": {"math": 1.0},
  "constraints": {"max_total_params_b": 2.0, "max_run_params_b": 2.0, "heal_steps_cap": 1000,
                  "compile_budget": 12},
  "objective": "max math score",
  "agent": {"model": "gpt-6-astra", "reasoning_effort": "high", "max_turns": 60, "max_tokens": 3000000}
}
```

`skills` maps each skill to its weight. The objective is the weighted mean
of the skill scores. A `null` constraint has no cap.

| Request | Skills | Total params cap | Compile budget |
|---|---|---|---|
| `math_plus` | math | 2B | 12 |
| `math_code_med` | math, code, medicine | 3.5B | 15 |
| `law_small` | law | 1.5B | 12 |
| `generalist_7b` | knowledge, math, code | 8B | 15 |

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
2. The constraint check estimates the total params and the params run per
   token from the atlas sizes. It also checks that heal, tap training, and
   each stitch are at or under the step cap (a stitch without `steps`
   counts as 500, the GraftSpec default), and that the program composes
   something. It rejects a block with a reversed or too deep layer range,
   a splice or tap layer past the model's last layer, a mask or task
   vector built on another model, a graft of a part type the compiler
   cannot graft, and an unknown heal, tap_train, or carve data name. A host with only a LoRA and a heal is the fine-tuning
   baseline, so the check rejects it. A model that is not in the atlas has
   no known size, so the check rejects it too.
3. Modal compiles the program with wsp on one H200 into
   `/vol/composer/<request>/<program hash>`, and scores it on the practice
   sets of the requested skills. It reports the measured total params, the
   params run per token, decode tokens per second at batch 1, the compile
   and practice seconds, a short compile report, and any errors.
4. The composer checks the measured size against the caps again. A program
   over a cap, or one whose compile did not report its size, cannot be
   submitted.

A compile costs one unit of the compile budget. A parse error, a
constraint rejection, or a program that already ran costs nothing. When the
budget is spent, the tool refuses and asks for `finish`. When the Modal
call itself fails (a timeout, a lost container), the compile still costs
its unit, but the result is not cached, so the same program can run again.

Each result has `practice_objective` (the practiced skills) and
`request_objective` (all request skills, an unpracticed skill as 0). The
loop ranks programs by `request_objective`, so a program practiced on one
easy skill does not win the fallback.

Params run per token count a tap donor's layers up to the tap's read layer,
because the donor runs on the prompt. That over-counts the decode cost of
taps, which keeps the check on the safe side.

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

## Results

`results/composer_<request>.json` holds:

- the final program (text and JSON), the rationale, and whether the loop
  chose it (`fallback`) because the model stopped without `finish`;
- its measured size and decode speed, and its practice scores;
- the budget used: compiles, turns, tokens, and the stop reason;
- held-out scores per set and per skill, and the objective, for the final
  program, the base alone, each source model the program used, and a
  LoRA on the base. The LoRA has rank 16, or the program's own LoRA
  settings when it has a LoRA. It trains on the program's heal data with
  the program's heal settings, for the sum of the program's heal,
  tap_train, and stitch steps, so it gets the same training budget. A
  program with no heal falls back to its tap_train data, then to the
  request's skill data (`composer_mix` for several skills);
- `fits_constraints` on every scored row: whether its measured size is
  under the total params cap. The base of a carve or a big donor is often
  over the cap, so it is a reference, not a rival;
- `heldout_overlap`: the parts the final program uses that were mined on
  held-out items (today, Person A's GSM8K parts).

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
