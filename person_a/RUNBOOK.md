# RUNBOOK: mining parts from open-weight models (for teammates' Claude agents)

You are a Claude Code agent helping one teammate on the weight-space programming project.
Your job: run the **parts-mining pipeline** on the models assigned to your teammate, then push
the parts it finds to GitHub. Read this whole file before you run anything.

> Status: v2 pipeline pushed. Load-tested so far: qwenmath1.5b, k2horizon7b, lfm2.5_8b_a1b. Other models are
> untested, so ALWAYS run `--smoke` first. Written against the v2 pipeline interface (`person_a/mine/`). If a command below
> fails because a file is missing, `git pull` first. If it is still missing, stop and tell your
> human; do not write your own replacement.

## 1. What the pipeline does (one paragraph)
For one donor model, an AI agent (OpenAI `gpt-6-astra`) reads the model's **raw weights** in a
locked-down sandbox and guesses what each part does, e.g. "layer 22, neurons 3010-3400: arithmetic".
This follows the "Sparks of Weight-Space Programming" technique. It can make up its own labels and
write a test for each one. A Modal GPU then **switches that part off** and scores the model on every
skill at once. A part is saved only if:
- switching it off clearly hurts its own skill,
- it barely touches the other skills,
- and it hurts more than random parts of the same size do.

Parts that pass go to `catalog/<model>.jsonl`, with an index file in `parts/<model>/`.

## 2. Who runs which model

| Person | Models (short names from `registry/donors.csv`) |
|---|---|
| Hanav (Person A) | `qwenmath1.5b`, `k2horizon7b`, `qwen3.6_35b_a3b`, `nanbeige33b` |
| Rex (Person B) | `qwen3.8_27b`, `gemma4_31b`, `phi4_14b`, `saul7b_inst`, `medgemma4b_inst` |
| Trisanth | `lfm2.5_8b_a1b`, `lfm2_24b_a2b`, `g9v3_39a5b`, `lfm2.5_1.2b_inst`, `lfm2.5_2.6b`, `lfm3_1.7b`, `lfm2.5_vl_450m`, `qwen3vl_8b`, `qwen3.5_9b`, `gemma4_26b_a4b`, `qwenmath7b_inst`, `mimo2.6_distill_9b`, `inkling_small`, `glm5.3_flash`, `glm5.3`, `deepseek_v4_flash`, `mimo2.6_flash`, `minimax_m3.1_flash`, `ling3.1_flash`, `mimo2.6_pro`, `mistral_large4` |
| Arjun | `k2mova36b`, `lfm2.5_24b_a2b`, `llama1b_host` |

Only run your own rows. If your human wants to swap, change this table in a commit first so two
people never run the same model.

## 3. One-time setup (about 10 minutes)
```bash
git clone https://github.com/trisanths/wsp.git && cd wsp/person_a
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/modal setup              # log in; use the shared team workspace if you have joined it
```

Secrets. Never paste a key into chat or a file that gets committed.
- **OpenAI key:** your human runs this in **their own terminal** (not through you). It hides the
  key while typing and keeps it out of shell history:
  ```bash
  printf "Paste OpenAI key: "; read -s k; echo "OPENAI_API_KEY=$k" > .env; chmod 600 .env; unset k; echo; echo saved
  ```
- **past.dev key (shared memory between all teammates' agents).** Use the team's shared `past_sk_...` key;
  get it privately from Hanav. Your human runs this in **their own terminal**:
  ```bash
  printf "Paste past.dev key: "; read -s k; echo "PAST_API_KEY=$k" >> .env; unset k; echo; echo saved
  ```
  Check it with `.venv/bin/python mine/shared_memory.py check`, which should end with `RESULT: OK`.
  With the key, every agent logs each attempt and lesson to the shared audience `wsp-mining` and
  searches it before proposing. Without the key, runs still work, but without shared memory.
- **Modal secrets** (once per Modal workspace; skip a step if `modal secret list` already shows it).
  The engine needs only `hf-token`, because the agent runs on your laptop:
  ```bash
  .venv/bin/modal secret create openai-key --from-dotenv .env
  .venv/bin/modal secret create hf-token HF_TOKEN=<hugging face token>   # human runs this one
  ```
  `.env` is in `.gitignore`. Check with `git check-ignore .env` before any commit.

Deploy the GPU engine (once per workspace, and again after pulling engine changes):
```bash
.venv/bin/modal deploy mine/engine_app.py
```

## 4. Check the setup first (cheap, about $1-3 per model)
```bash
.venv/bin/python mine/run.py --model <short_name> --smoke
```
This loads the model on a GPU, prints its structure (layers, heads, MLP width, experts) and its
base score on each built-in skill, then does one switch-off test. **No agent, no OpenAI cost.**
If it fails, stop and report the error to your human. Don't patch engine code yourself unless
your human says so. Load failures on custom-code models (`k2*`, `g9v3*`) are the most likely kind.

## 5. Run the mining
One model:
```bash
.venv/bin/python mine/run.py --model <short_name> --budget-usd 25 --max-hours 3 --person <your name>
```
All your models in parallel:
```bash
.venv/bin/python mine/run_many.py --models <a>,<b>,<c> --budget-usd 25 --max-hours 3 --person <your name>
```
- `--budget-usd` caps OpenAI spend **per model**. Modal GPU time is extra; the run log prints both.
- `--agents 3` runs 3 agents on each model at once, each with a different focus (attention heads, early MLP
  neurons, late MLP neurons, experts/blocks). They share findings through the memory. This is faster, and each agent
  has its own `--budget-usd`, so lower that (e.g. `--agents 3 --budget-usd 10`).
- `--person` labels your entries in the shared memory, so other agents can see who tried what.
- To cut OpenAI cost, add `--effort medium`. OpenAI is most of the cost: about $8 for a high-effort run
  on the small model, against about $1 of Modal.
- To stop a run, use **Ctrl-C** (or `kill -INT`), never `kill -9`, so it shuts down its Modal sandbox.
- Runs can be resumed: if one stops, run the same command again and it continues where it left off.
- Run long jobs in the background and report progress to your human at milestones, not constantly.

## 6. What you get
- `catalog/<short_name>.jsonl`: one line per part. The fields are:
  - `id`, `type` (`mlp_cluster` | `attn_heads` | `expert` | `block`)
  - `label`, `tier` (`screened`, or `confirmed` if the optional stricter check passed)
  - `model`, `layer`/`layers`, `units` (path to the `.pt` index file), `d_model`
  - `signature` (score drops per skill)
  - whether the label was built-in or agent-defined, and the agent's weight rationale
  - in `signature`: `z` (the target drop on the search half as a z-score against the random same-size
    parts: the drop minus their mean, divided by their sample standard deviation; null with fewer than
    2 random drops or when they are all equal), `random_n`, `random_mean`, `random_std`. Read `z` as a
    rough number: it is from the search half, the half the part was selected on, and from the gate's
    5 random parts (`GATE_N_RANDOM` in `mine/client.py`). The engine draws them with replacement and
    rounds each score, so drops can repeat: `random_distinct` counts the distinct values, and `z_flag`
    is set when there are fewer than 3.
  - in `signature`: `z_heldout` and `random_heldout` (the same for the held-out half; null while the
    engine scores random parts on the search half only)
  - in `signature`: `guard` (for `search` and for `heldout`: the skill, other than the target and its
    own control, whose score moved most when the part was off, as `skill`, `drop`, `abs`)
  - `selection_data` (the data the part was selected on): `skill`, `source` (`agent` | `builtin` |
    `pool`), `path` and `hash` of the frozen test for an agent skill, `description`, `datasets`,
    `control`, `off_target` (the other skills the gate checked) with `off_target_data`, `pool` (the
    text pool of `wsp/healdata.py` that matches the skill, or null) with `pool_match`, and
    `reported_benchmarks` (the dataset ids in the selection data that are in the `BANNED` list of
    `wsp/healdata.py`; null means not checked). `builtin_basis` says where the built-in
    descriptions came from: `engine`, `local file`, `git history`, or
    `git history, version uncertain`. `items_sha` names the version of a built-in set.
    `selection_data` is null, with the reason in `selection_note`, when the files behind it could
    not be read at the time the row was written; `mine/enrich_rows.py` fills it in.
  - in `selection_data`, for an agent skill: `path_matches` (does the file at `path` still have the
    row's `hash`? Lanes on one model share `mine/skills/<short_name>/`, and a later lane can replace
    a test of the same name; `path_hash` and `path_note` say what was found). An off-target agent
    skill has its `hash`, or null and `hash_candidates` when more than one test of that name is known.
  - in `selection_data`, with `git history, version uncertain`: the backfill could not tell which
    version of the built-in file the engine held for the row. `reported_benchmarks` is then the
    union over the versions, `builtin_candidates` lists what each version says, and `builtin_note`
    says how to settle it.
  - `routing` (expert parts only): for each expert, `skill_rate` and `general_rate` (router picks per
    token on the target skill's text and on `general_text`) and their `log_ratio`, for `search` and
    `heldout`, and the `revision` of the weights the picks were counted on. It is null, with the
    reason in `routing_note`, when the engine cannot count the picks (`module_list` experts, router
    indices that do not fit the batch, a target skill the engine does not hold, or a deployment
    older than `expert_pick_rates`). A note that starts with `no routing, final:` names a reason
    that another engine call cannot change.
  - `enriched_at`: set on rows that `mine/enrich_rows.py` filled in after they were written
- `parts/<short_name>/*.pt`: index files `{kind, model, layer, indices}`.
- `mine/skills/<short_name>/`: the tests the agent wrote for its own labels. These are frozen and hashed.
- `logs/<short_name>/`: the run transcript, experiment log and cost.

Rows written before these fields existed get them from the stored numbers, with no GPU call:
```bash
.venv/bin/python mine/enrich_rows.py --model <short_name>      # repeat --model for more models
```
It keeps every field and the row order, and it never touches `*.rejected.jsonl`.
- Run it on the machine that mines the model, after the run has stopped and the catalog is pushed.
  It rewrites every line of the file, so a teammate who appends rows to the same file on another
  machine gets a merge conflict.
- It refuses a model whose `logs/<short_name>/` changed in the last 10 minutes (a live run), and a
  model that has no `logs/<short_name>/` folder on this machine (the run can be on another machine),
  unless you pass `--force`. A fresh `git pull` or checkout also gives the log files new times, so
  wait 10 minutes after one.
- Do not run it while an agent run or `mine/regate.py` works on the model, also not with `--force`.
  The backfill holds the catalog lock against new rows, but the agent's `confirm_math` tool and
  `regate.py` replace the file without that lock and can write the old rows back.
- For rows without an engine-reported description, the built-in sets come from the git history of
  `mine/data/builtin_skills.json`. The engine uses the file from its last deploy, and git does not
  hold the deploy time. A row made after the commit of a newer version gets
  `git history, version uncertain`. Ask the person who deployed when the engine began to use that
  version, and pass `--live-since <commit>=<YYYY-MM-DDTHH:MM:SS>` (local time).
- `--routing` also fills `routing` for expert rows; that one flag calls the GPU engine, and it
  prints the number of engine calls first. The engine is built for the `revision` of each row. A row
  with a `no routing, final:` note is not asked again unless you add `--retry-routing`.

## 7. Push your results (no merge conflicts by design)
Every model writes only its **own** files, so teammates never edit the same file.
```bash
git pull --rebase origin main
git add catalog/<short_name>.jsonl parts/<short_name>/ mine/skills/<short_name>/ logs/<short_name>/
git status            # make sure ONLY your models' files are staged; never .env, never .venv
git commit -m "Parts for <short_name>: <N> parts (<labels>)"
git push origin main
```
Push after each model finishes; don't wait for all of them. If the push is rejected, run
`git pull --rebase origin main` and push again.

## 8. Building the combined catalog
**Only Hanav (or whoever your human names) does this**, so the combined file has one writer:
```bash
.venv/bin/python mine/compile_catalog.py   # merges catalog/*.jsonl -> catalog.jsonl
```
Then commit and push `catalog.jsonl`. This is the file Trisanth's `compile()` reads.

## 9. Rules
- Never invent numbers. If something fails, report the exact error.
- Never touch another person's model files, Modal apps or running processes. No `pkill` or
  `killall` by name.
- Never commit secrets. Never print keys.
- Don't delete things in the shared Modal volume `swarmy-models`.
- Stay inside the budget. If one model looks like it will cost more than about $40 in total, stop
  and ask your human.
- When a run finishes, give your human a 3-line summary:
  - how many parts and which labels,
  - cost (Modal + OpenAI),
  - anything that broke.

## 10. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `Secret 'openai-key' not found` | Secret not created in this Modal workspace | Section 3 |
| `401` from OpenAI | Wrong key, e.g. an old key in `~/.zshrc` | The scripts read `.env` with override; check `.env` |
| Out of GPU memory | Model is larger than the chosen GPU | Report it; the engine picks the GPU by model size, so this is a bug |
| `trust_remote_code` / unknown architecture | Custom-code model (`k2*`, `g9v3*`) | Report it to your human with the traceback |
| Smoke shows a skill at chance level | The model can't do that skill | Normal: the skill is marked `skip` for this model |
| Old behaviour after `modal deploy` | Warm containers still run old code | `modal container list`, then `modal container stop <id>` for that app |
