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

## 8a. Skill maps (model level, no agent)
`mine/maps.py` measures where each skill sits in one model. It needs no agent and no OpenAI key.
```bash
.venv/bin/python mine/maps.py --zoo-key <key in atlas/zoo.json>
.venv/bin/python mine/maps.py --zoo-key <key> --modules
.venv/bin/python mine/maps_many.py --zoo-keys A,B,C --jobs 3 --modules
```
What it measures. The engine skips each decoder layer in turn and scores every skill set. The
score of a layer for a skill is the loss increase on the skill text minus the loss increase on
`general_text`, in nats. Blocks are the shortest windows of layers that hold 60% of the positive
score. They are chosen on the search half of each set and checked on the held-out half. For
layers with fused experts, the engine also counts how often the router picks each expert on each
skill's text.

What `--modules` adds. For each layer, the engine also scores every skill set with only the
attention output zeroed, and with only the feed-forward output zeroed. The feed-forward module is
the dense MLP, the routed experts, or both when a layer has a shared expert next to routed
experts. The parts zero the input of the out-projection, so a bias on that projection stays.

Cost in scoring passes. One pass scores every skill set once (39 sets with the current pool file,
48 search items and 24 held-out items each).
- A plain run takes one pass for the base scores and one pass per layer. A model with fused
  experts also gets one forward pass over each half of each skill set, on at most 64 items per
  half, for the router counts.
- `--modules` adds one pass per layer that has an attention module and one pass per layer that
  has a feed-forward module. That is up to three passes per layer in place of one.
- One engine call maps the whole model. The call timeout in `engine_app.py` is 3600 s on the
  single-GPU tiers and 3 hours on the multi-GPU tiers. Read `engine_seconds` of a plain entry of
  the same model before you start a `--modules` run.
- `--modules` needs an engine that was deployed with `skill_maps(modules=...)`. A plain run sends
  the same call as before, so it also works with an older deployed engine.
- An entry that exists is skipped. With `--modules`, an entry without `module_maps` is measured
  again and replaced. `--force` always measures again.

Output: `atlas/pool_entries/<org>__<name>.json`. Fields per skill:
- `layer_maps[skill]`: `scores` and `influence` (one number per layer), `blocks`, `block_mass`,
  and `heldout` with `check_mass` and `split_half_r`.
- `layer_maps[skill].block_z`: for each block, its positive score mass against the positive mass
  of every other window of the same length in the same model, in sample standard deviations. The
  other windows may overlap the block. The value is null when there are fewer than 3 other
  windows or when they all hold the same mass. `block_z` uses the search scores, which also chose
  the block. `block_z_heldout` uses the held-out scores for the same block.
- `module_maps[skill]` (only with `--modules`): `attn` and `mlp`, the layer score definition for
  one module, with null where the layer lacks the module. `module_share` is the attention share
  of the positive `attn` + `mlp` score per layer. A missing module counts as 0, and the share is
  null when neither score is positive. `heldout.split_half_r.attn` and `.mlp` are the Pearson r
  of the search and held-out score lists.
- `module_parts` (only with `--modules`): per layer, the part kinds that were zeroed for `attn`
  and for `mlp`. An empty list means the layer lacks the module.
- `expert_maps[skill]` and `expert_details[skill]`: per layer, the experts that the router picks
  more on skill text than on general text, on both halves.
- `selection_data[skill]`: one string with the dataset and the description of the skill text.
- `selection_sources[skill]`: `dataset`, `description`, `pool` (the `wsp/healdata.py` pool that
  draws on the same dataset, or null), and `reported_benchmark` (true when the dataset id is in
  the `BANNED` list of `wsp/healdata.py`). Mining may use benchmark data, so read this field
  before you train or select on a skill's text. `benchmark_free` is written as true for every
  entry and does not look at the skill list.

Tests (CPU only, no Modal call): `.venv/bin/python mine/tests/test_module_parts.py`,
`mine/tests/test_skill_maps_engine.py`, and `mine/tests/test_module_maps.py`. Each prints `ok`.

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
