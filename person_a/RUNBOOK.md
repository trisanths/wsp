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
such a member against the host and against the ancestor. The file has 6 families and 23 pairs.

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

Commands (from `person_a/`, in the Modal workspace that has the volume `swarmy-models`):
```bash
.venv/bin/modal run mine/weight_delta.py --family lfm2          # one family
.venv/bin/modal run mine/weight_delta.py --all --jobs 4         # all 23 pairs, 4 at a time
.venv/bin/modal run mine/weight_delta.py --member LiquidAI/LFM2-1.2B-Tool --reference LiquidAI/LFM2-1.2B
.venv/bin/python mine/tests/test_weight_delta.py                # local tests, must print "ok"
```
- Use `modal run` only. The app `swarmy-delta` is ephemeral. Do not run `modal deploy` on it.
- `--jobs` is the largest number of containers that run at the same time (default 4).
- A pair whose output file exists is skipped. Add `--force` to measure it again.
- Step 1 downloads each model once (`*.safetensors` and `*.json` of the pinned revision) into
  the volume cache `/models/hf_cache`, one container per model. Step 2 starts one container
  per pair (8 CPU, 8 GB) and reads the cache. The job reads tensors in slices of at most
  16.8 million elements and never loads a whole model.
- Measured on LiquidAI/LFM2-1.2B-Tool against LiquidAI/LFM2-1.2B (2.34 GB each, 2026-11-14):
  4.8 s to compare, peak process memory 1.09 GB. Larger pairs have not been run.

Output: `atlas/lineage/weight_delta/<member key>__vs__<reference key>.json`, where a key is
the Hugging Face id with `/` replaced by `__`. The fields are:
- `member`, `reference`, `relation` (`host` or `ancestor`), `family`, `host`
- `member_revision`, `reference_revision`: the pinned commits
- `layers`: the number of decoder layers of the member; `kinds`: the module kinds found
- `per_layer`: one object per layer, `{"layer": N, "attn": {...}, "mlp": {...}, "norm": {...}}`.
  Each group has `rel`, `cosine`, `tensors`, and `params`.
- `outside_layers`: the groups `embed`, `lm_head`, `norm`, `other`. A group is `null` when no
  tensor of that kind matched (for example a tied `lm_head` that the checkpoint does not store).
- `summary`: `all` (every compared tensor) and `layers` (one group per module kind over all layers)
- `tensors`: one row per compared tensor: `name`, `shape`, `layer`, `kind`, `rel`, `cosine`,
  `diff_norm`, `reference_norm`, and `reference_name` when the reference uses another name
- `unmatched_member`, `unmatched_reference`: tensors that only one model has
- `shape_mismatch`: tensors with the same name and different shapes. When only the number of
  rows differs (an embedding with added tokens), `shared_rows` has `rel` and `cosine` over the
  rows that both models have. These tensors are not part of any group value.
- `counts`: totals for the lists above and `layer_match_fraction`
- `dtype` (`float64`, the arithmetic), `source_dtypes` (the stored dtypes of each model)
- `weights_gb`, `compute_seconds`, `peak_memory_gb`, `container`, `measured_at` (UTC)

For each pair the job also prints, per module kind, the 5 layers with the largest `rel`.
