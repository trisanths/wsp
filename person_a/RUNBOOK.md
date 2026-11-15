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
| Rex (Person B) | `qwen3.8_27b`, `gemma4_31b`, `phi4_14b`, `saul7b_inst` |
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
- A plain run takes one pass for the base scores and one pass per layer, in one engine call. A
  model with fused experts also gets one forward pass over each half of each skill set, on at
  most 64 items per half, for the router counts.
- `--modules` adds one pass per layer that has an attention module and one pass per layer that
  has a feed-forward module. That is up to three passes per layer in place of one.

How a `--modules` run works. An engine call that reaches its timeout returns nothing (3600 s on
the single-GPU tiers, 3 hours on the multi-GPU tiers, in `engine_app.py`). So the module passes
do not go into the plain call.
1. The layer and expert maps are kept from the entry that is there, when it is for the same model
   revision and the same skill text. The whole-layer passes are not paid again. When there is no
   such entry, the plain call runs first and its entry is written before any module call.
2. The module passes go to the engine in calls of a few layers each. A call for N layers takes at
   most 1 + 2 x N passes. `maps.py` takes the seconds per pass from the plain entry
   (`engine_seconds` over 1 + `layers` passes) and picks the most layers for which that estimate
   stays within half of the 3600 s timeout. `--layers-per-call N` sets the number by hand. When
   the estimate for the 3 passes of one layer is above 3600 s, the run stops before any call.
3. After each call the merged result is saved in
   `atlas/pool_logs/<org>__<name>.modules-partial.json`. A run that fails starts again after the
   saved layers, when the revision and the skill sets are the same. The file is removed when the
   entry is written. `--force` does not use it.
- `--modules` needs an engine that was deployed with `skill_maps(modules=, layers=, plain=)`.
  With an older engine `maps.py` prints that and exits with code 3, and `maps_many.py` starts no
  more models. A plain run sends the same call as before, so it works with an older engine.
- An entry that exists is skipped. With `--modules`, an entry is skipped when it has the
  `module_maps` field. `--force` always measures again, the plain passes included.
- A plain run does not replace an entry that has `module_maps` unless `--force` is set.
- `maps_many.py --modules` writes `atlas/pool_logs/<key>.modules-failed` when the run of a model
  fails, and does not start that model again in later `--modules` runs. `--retry-failed` starts
  it again. A run that ends well removes the file. Read `atlas/pool_logs/<key>.log` first.

Output: `atlas/pool_entries/<org>__<name>.json`. Fields per skill:
- `layer_maps[skill]`: `scores` and `influence` (one number per layer), `blocks`, `block_mass`,
  and `heldout` with `check_mass` and `split_half_r`.
- `layer_maps[skill].block_z`: for each block, its positive score mass against the positive mass
  of every other window of the same length in the same model, in sample standard deviations. The
  value is null when there are fewer than 3 other windows or when they all hold the same mass.
  `block_z` uses the search scores, which also chose the block. `block_z_heldout` uses the
  held-out scores for the same block.
- How to read `block_z`. It is not a test against independent random blocks. Most of the other
  windows share layers with the block: `block_z_windows` is the number of other windows and
  `block_z_disjoint_windows` is how many of them share no layer with it. The first block is the
  window with the most mass on the search scores, so its `block_z` is positive or null by
  construction. Only `block_z_heldout` and `block_z_disjoint_heldout` are a check. A short block
  is compared with windows that mostly hold zero mass, so the deviation is small and z can be
  very large. Do not read it as a significance level.
- `layer_maps[skill].block_z_disjoint` and `block_z_disjoint_heldout`: the same z against only
  the windows that share no layer with the block. Null with fewer than 3 such windows or when
  they all hold the same mass.
- `module_maps[skill]` (only with `--modules`): `attn` and `mlp`, the layer score definition for
  one module, with null where the layer lacks the module. `module_share` is the attention share
  of the positive `attn` + `mlp` score per layer. It is null where the layer lacks one of the two
  modules (the `attn` and `mlp` lists still have the score that was measured), and null when
  neither score is positive. A score at or below 0.00002 nats counts as 0 in the share, because
  the four rounded log-probs behind a score can move it by that much. The share has no other
  floor: read it with the two scores. `heldout.split_half_r.attn` and `.mlp` are the Pearson r of
  the search and held-out score lists over the layers that have the module, and
  `heldout.n_layers.attn` and `.mlp` are how many layers that is. r is null below 4 layers.
- The adapter knows two modules per layer. Where it finds no attention module (the conv layers
  of the LFM2 models), `attn` and `module_share` are null and the operator that sits there is
  not measured.
- `module_parts` (only with `--modules`): per layer, the part kinds that were zeroed for `attn`
  and for `mlp`. An empty list means the layer lacks the module.
- `module_engine_seconds`, `module_calls`, `module_measured_at` (only with `--modules`): the
  engine time and the number of the module calls, and when the module fields were written.
  `engine_seconds` and `measured_at` stay those of the plain call.
- `expert_maps[skill]` and `expert_details[skill]`: per layer, the experts that the router picks
  more on skill text than on general text, on both halves.
- `selection_data[skill]`: one string with the dataset and the description of the skill text.
- `selection_sources[skill]`: `dataset`, `description`, `pool` (the `wsp/healdata.py` pool that
  draws on the same dataset, or null), and `reported_benchmark` (true when the dataset id is in
  the `BANNED` list of `wsp/healdata.py`). Mining may use benchmark data, so read this field
  before you train or select on a skill's text.
- `reported_benchmark_skills`: the names of the skills with `reported_benchmark` true, or null
  when `wsp/healdata.py` was not found. `benchmark_free` is written as true for every entry, as
  before, and does not look at the skill list. An entry can have `benchmark_free: true` and a
  list of reported skills here: trust the list. The owner decides if `benchmark_free` goes away.
- A plain entry that `--modules` extends keeps its layer fields as they were written. An entry
  from before `block_z` existed gets `block_z` only from a `--force` run.

Tests (CPU only, no Modal call), each prints `ok`:
```bash
for t in mine/tests/test_*.py; do .venv/bin/python $t; done
```

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

## 11. Weight-difference maps (fine-tune families)
The weave step of the compiler mixes a fine-tune with its host model per layer and per module
kind. It adds (member - ancestor) weight differences onto the host, so it must know where each
fine-tune differs from its host. `mine/weight_delta.py` measures that from the weights alone.
It uses CPU containers and no GPU, no agent, and no LLM.

The families are in `atlas/lineage/families.json`: a short name, the host, and the members.
A member with an `ancestor` was tuned from that model and not from the host. The job measures
such a member against the host and against the ancestor. The file has 6 families and 26 pairs.
The ancestor `unsloth/Meta-Llama-3.1-8B` is not a model of `atlas/zoo.json`. Every host and
every member is.

What the job measures, for one pair (member, reference):
- For each tensor that the two checkpoints have with the same shape: `rel`, which is
  ||W_member - W_reference|| / ||W_reference|| (Frobenius norm), and `cosine`, the cosine of the
  two flattened tensors. The job casts the two tensors to float64 before it subtracts them.
- The same two numbers for each decoder layer and module kind. The kinds are `attn` (attention
  projections), `mlp` (feed-forward projections), `norm` (norm weights, also the q and k norms
  inside the attention module), and the module name for any other mixer (LFM2 has `conv`).
  A group value is sqrt(sum of squared differences) / sqrt(sum of squared reference norms)
  over the tensors of the group.
- The same two numbers outside the layers: `embed`, `lm_head`, `norm` (final norm, embedding
  norm), and `other`.

Names are matched after the wrapper prefixes are removed (`model.`, `language_model.`), so a
text-only member matches a multimodal host. A tensor is a decoder-layer tensor only when its
name starts with `layers.<N>.` after that step. Vision tower layers stay outside. If fewer than
90% of the layer tensors of the member match the reference in name and shape, the pair fails
with a message and no file is written.

A pair also fails, with a message and no file, in these cases:
- The decoder layers of the member or of the reference are not numbered 0 to N-1 with none
  missing, or the two models do not have the same number of layers.
- A folder has no `model.safetensors.index.json` and fewer `model-XXXXX-of-YYYYY.safetensors`
  files than YYYYY.
- A compared tensor holds a NaN or an infinite value. The message names the tensor.

Commands (from `person_a/`, in the Modal workspace that has the volume `swarmy-models`):
```bash
.venv/bin/modal run mine/weight_delta.py --family lfm2          # one family
.venv/bin/modal run mine/weight_delta.py --all --jobs 4         # all 26 pairs, 4 at a time
.venv/bin/modal run mine/weight_delta.py --member LiquidAI/LFM2-1.2B-Tool --reference LiquidAI/LFM2-1.2B
.venv/bin/python mine/tests/test_weight_delta.py                # local tests, must print "ok"
```
- Use `modal run` only. The app `swarmy-delta` is ephemeral. Do not run `modal deploy` on it.
- `--jobs` is the largest number of containers that run at the same time (default 4).
- A pair whose output file exists is skipped. Add `--force` to measure it again.
- A model whose revision lookup or download fails does not stop the run. Its pairs are listed
  as `FAILED`, the other pairs are measured, and the command exits with code 1. Run the same
  command again to measure only the pairs that have no file.
- Step 1 downloads each model once (`*.safetensors` and `*.json` of the pinned revision) into
  the volume cache `/models/hf_cache`, one container per model. Step 2 starts one container
  per pair (8 CPU, 8 GB) and reads the cache. The job reads tensors in slices of at most
  16.8 million elements and never loads a whole model.
- `--all` downloads every model before it measures the first pair (27 models when the size below was
  taken; the Llama-3-8B-Instruct reference for Med42 was added later and is not in that figure). The files that match the
  two patterns add up to 800.3 GB (Hub file listings, 2026-11-14). Each of the 7 Qwen 32B
  models is 65.5 GB, and the two Gemma models are 54.9 GB and 54.05 GB.
- Measured on LiquidAI/LFM2-1.2B-Tool against LiquidAI/LFM2-1.2B (2.34 GB each, 2026-11-14):
  4.8 s to compare, peak process memory 1.09 GB. Larger pairs have not been run.

Output: `atlas/lineage/weight_delta/<member key>__vs__<reference key>.json`, where a key is
the Hugging Face id with `/` replaced by `__`. The fields are:
- `member`, `reference`, `relation` (`host` or `ancestor`), `family`, `host`
- `member_revision`, `reference_revision`: the pinned commits
- `layers`: the number of decoder layers of the member; `kinds`: the module kinds found
- `per_layer`: one object per layer, `{"layer": N, "attn": {...}, "mlp": {...}, "norm": {...}}`.
  Each group has `rel`, `cosine`, `tensors`, and `params`. A layer object has a kind only when
  the layer has tensors of that kind, so read it with `entry.get("attn")`. In LFM2-1.2B, layers
  2, 5, 8, 10, 12, and 14 have `attn` and the other 10 layers have `conv`.
- `outside_layers`: the groups `embed`, `lm_head`, `norm`, `other`. A group is `null` when
  neither model stores a tensor of that kind with a partner in the other model (for example a
  tied `lm_head` that the checkpoint does not store).
- A group with a `shape_mismatch` list (outside the layers or in a layer) has tensors that the
  two models store with different shapes. Each item has `name`, `member_shape`,
  `reference_shape`, and `shared_rows` (`rows`, `rel`, `cosine` over the rows that both models
  have, or `null` when another dimension differs). `rel` and `cosine` of the group cover only
  its tensors of equal shape. They are `null`, with `tensors` 0, when the group has none: an
  embedding with added tokens gives `"embed": {"rel": null, ..., "shape_mismatch": [...]}`.
- `summary`: `all` (every compared tensor) and `layers` (one group per module kind over all layers)
- `tensors`: one row per compared tensor: `name`, `shape`, `layer`, `kind`, `rel`, `cosine`,
  `diff_norm`, `reference_norm`, and `reference_name` when the reference uses another name
- `unmatched_member`, `unmatched_reference`: tensors that only one model has
- `shape_mismatch`: tensors with the same name and different shapes. When only the number of
  rows differs (an embedding with added tokens), `shared_rows` has `rel` and `cosine` over the
  rows that both models have. These tensors are not part of any group value and not part of
  `summary`. Their groups list them (see above).
- `counts`: totals for the lists above and `layer_match_fraction`
- `dtype` (`float64`, the arithmetic), `source_dtypes` (the stored dtypes of each model)
- `weights_gb`, `compute_seconds`, `peak_memory_gb`, `container`, `measured_at` (UTC).
  `peak_memory_gb` is the maximum of the container process. Modal reuses a container for a
  later pair, so the number can come from an earlier, larger pair of the same run.

For each pair the job also prints, per module kind, the 5 layers with the largest `rel`, and
one line for the tensors outside the layers. That line shows the shared-row `rel` of a
resized embedding.
