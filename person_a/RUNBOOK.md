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

## 5b. Sweep miner (no LLM, large parts)
`mine/sweep.py` mines large parts for many skills on one model. It uses no agent and no OpenAI key.
The compiler asked for larger units than the agents find: MLP clusters of 64 units and more, groups of
heads, expert sets, and layer blocks.

What it does:
1. It checks which skills the model can do (base score against shuffled answers) and skips the others.
2. For each skill and each unit kind the model has, it asks the engine for the best units of every layer
   by attribution patching: the estimated loss rise on the skill when a unit is zeroed, minus the same
   on `general_text`. A quantized (FP8) model cannot run this, so the sweep ranks by activation there.
   Experts that a model keeps as a list of modules are ranked by attribution only.
3. It builds candidates in the 3 best layers: MLP clusters of 64, 256, and 1024 units, head groups of 4
   and 8, and expert sets of 2, 4, and 8. It adds the layer blocks of
   `atlas/pool_entries/<org>__<name>.json` when `mine/maps.py` has written that file.
4. It screens the candidates in order, best first, with the same switch-off test and light gate as the
   agents. A part is saved when the light gate passes and its drop is at least 2 standard deviations
   above the drops of the random parts of the same size (z-score, search half).

The order. A summed score always puts the largest part of a layer first, and a score per unit always
the smallest. So the sweep sorts on a score that does not depend on the size: the summed score of a
candidate divided by the mean summed score of all candidates of the same kind and size in this run. The
order tells you which candidates stand out among parts of their own size. It does not tell you which
size is best.

The z-score. It follows the rule of the row field `signature.z`, which the branch `mining/row-evidence`
adds (mean and sample standard deviation of the random drops). The engine rounds drops to 4 decimals,
so the random drops of a small part can all be equal. Then no deviation is measured and there is no
z-score: the row has `z_search: null` and `random_spread: "below resolution"`, and the part is saved
only when its drop is at least 0.15 (the lowest target drop of the light gate) above the random drops.

```bash
.venv/bin/python mine/sweep.py --short <short_name> --skills law,medicine,math
.venv/bin/python mine/sweep.py --short <short_name> --skills all --max-screens 200 --max-hours 4
.venv/bin/python mine/sweep.py --short <short_name> --skills law --rank-only     # print the plan, screen nothing
```

Skills are the built-in sets (`math`, `code`, `facts`, `multilingual`) and the sets in
`mine/data/pool_skills.json`. Each screen scores the target, `general_text`, and 5 guard skills from
other domains (the list is at the top of `mine/sweep.py`), not every skill set. When the model cannot do
a guard skill, another skill of the same domain takes its place. The summary names every skill that was
screened with fewer than 5 guards. A skill with fewer than `--min-guards` guards is skipped.

| Option | Default | Meaning |
|---|---|---|
| `--skills` | required | comma-separated names, or `all` |
| `--max-screens` | 60 | new switch-off tests in this run; finished ones are free |
| `--max-hours` | 3 | stop starting new work after this time |
| `--kinds` | all four | any of `mlp_cluster`, `attn_heads`, `expert`, `block` |
| `--mlp-sizes`, `--head-sizes`, `--expert-sizes` | `64,256,1024`, `4,8`, `2,4,8` | part sizes |
| `--top-layers` | 3 | layers per skill and kind |
| `--rank-items` | 24 | search items per ranking pass |
| `--z-min` | 2.0 | lowest z-score that passes |
| `--min-margin` | 0.25 | skip a skill when the base margin over shuffled answers is lower (nats) |
| `--min-guards` | 3 | skip a skill when fewer guard skills than this can be scored |
| `--order` | `global` | `global`: highest score first over all skills and kinds; `round`: the best candidate of every skill and kind first, then the second best |
| `--lane` | empty | a name for a second sweep on the same model at the same time |
| `--rank-only` | off | rank and print the plan only |

- The sweep can be resumed. Run the same command again: rankings and finished screens are kept, and
  `--max-screens` counts only new screens. With `--skills all`, raise `--max-screens`; the default
  covers only the candidates with the highest scores.
- A rerun does not write a row again. If you moved a row out of the catalog, it stays out. A part whose
  row is in `catalog/<short_name>.rejected.jsonl` is never written. A kept screen is checked again with
  the light gate of today.
- A ranking that fails (for example out of memory) is named in the summary, and the run goes on with
  the other skills and kinds. The next run tries that ranking again. 3 failures in a row stop the ranking.
- It is safe to run next to an agent run on the same model. The sweep keeps its state in
  `logs/<short_name>/state.lanesweep.json`, and it appends to the catalog under the same lock as the agents.
  It takes the pinned revision from the agents' state (`state.json`, or the newest `state.lane<N>.json`)
  and prints which file it used. If the lane files pin different revisions, the sweep stops: tell your human.
- To rank again from nothing, move `logs/<short_name>/sweep/` away. To screen again, also move the
  lane state file away.
- The deployed engine needs the methods `attribution_units` and `activation_units`. If the run says a
  method is missing, the engine was deployed from an older branch: tell your human.

Output:
- Rows in `catalog/<short_name>.jsonl` with tier `screened`, a label such as
  `law, MLP cluster of 256 units, layer 14 (attribution-ranked)`, and `evidence_used`
  `["attribution", "ablation"]` (`["activation", "ablation"]` for a quantized model, `["ablation"]`
  for a block; a block is ranked by the layer map, which is itself an ablation). Each row has one more
  field, `sweep`: the ranking method and summed score, the z-score (or the gap, when the random drops
  have no measured spread), and the guard skills. For a pool skill, `skill.source` is `pool` and the row
  names the dataset.
- `logs/<short_name>/sweep/results.jsonl`: one line per screen, passed or not, with the reasons.
- `logs/<short_name>/sweep/rank_<skill>_<kind>.json`: the candidates of each ranking.
- A summary at the end: candidates ranked, screened, and passed, by kind and by skill, then notes:
  skipped skills, short guard sets, failed rankings, and the parts that passed.

Push the results as in section 7.

## 6. What you get
- `catalog/<short_name>.jsonl`: one line per part. The fields are:
  - `id`, `type` (`mlp_cluster` | `attn_heads` | `expert` | `block`)
  - `label`, `tier` (`screened`, or `confirmed` if the optional stricter check passed)
  - `model`, `layer`/`layers`, `units` (path to the `.pt` index file), `d_model`
  - `signature` (score drops per skill)
  - whether the label was built-in or agent-defined, and the agent's weight rationale
- `parts/<short_name>/*.pt`: index files `{kind, model, layer, indices}`.
- `mine/skills/<short_name>/`: the tests the agent wrote for its own labels. These are frozen and hashed.
- `logs/<short_name>/`: the run transcript, experiment log and cost.

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
