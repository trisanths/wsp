# The skill atlas

The atlas answers two questions for the compiler and the planner. Which
open model is strongest at a skill? And inside that model, which layers
(or, for MoE models, which experts) carry the skill? With those answers a
program can pick donors and donor blocks from measurements instead of
guesses.

Code: `wsp/atlas.py`. Model list: `atlas/zoo.json`. Runner:
`modal_atlas.py`. Entries: `atlas/entries/<org>__<name>.json`.

## The zoo

`atlas/zoo.json` lists 113 models: Liquid (LFM2, LFM2.5, LFM3, and the d1
decision models), legal (Saul 7B and 54B, law-chat, two Chinese legal
models), math (Qwen2.5-Math, the DeepSeek-R1 distills, NuminaMath,
deepseek-math, OpenMath2, Phi-4-mini-reasoning), code (Qwen2.5-Coder 0.5B
to 14B, deepseek-coder), medical (OpenBioLLM, BioMistral, Meditron, Med42,
HuatuoGPT-o1, II-Medical, MedGemma), general models from Qwen, Llama,
Gemma, Phi, Mistral, OLMo-2, Granite, SmolLM and Nanbeige, and MoE models
from Qwen3-30B-A3B up to DeepSeek-V4-Pro, GLM-5.3-Flash and Kimi-K3.

Each entry has the Hugging Face id, maker, model type, size, layer count,
hidden size, kind (base, instruct, reasoning, specialist), the skills the
model is known for, the license, and what it needs to run: the
transformers version, trust_remote_code, and whether it needs several GPUs.
Every id was checked against the Hugging Face API on 2026-11-14.
AI-MO/NuminaMath-7B-TIR moved to project-numina/NuminaMath-7B-TIR, and
the zoo uses the new id. One model
is gated (epfl-llm/meditron-7b, automatic approval). Gated Google and Meta
repos are replaced by their ungated unsloth mirrors.

`needs.transformers` is 4.57.1 when the model type is native in
transformers 4.57.1 (or the repo's own code was written for 4.x), else
5.x. In the 4.57.1 image, 86 models run. These are skipped with a reason:
LFM2 MoE models, LFM3 and the d1 models, Qwen3.5 and Qwen3.8, Gemma 4,
K2-Horizon, and the four giants. The giants (more than 150B parameters)
are marked `multi_gpu` and `expert_level_only`. The `--tf5` run (see Run)
covers the 22 skipped models that are not giants, on a transformers 5.23.0
image. DeepSeek-V4-Pro (about
1.6T parameters) and Kimi-K3 (about 2.8T) do not fit on H200:8 in bf16.

## What each screen measures

Each skill has a screen and a scorer (`SKILLS` in `wsp/atlas.py`):

| Skill | Screen | Scorer |
|---|---|---|
| knowledge | MMLU test, round 6 dev sample (1000) | letter log-probability |
| medicine | MedMCQA validation, round 10 order (1000) | letter log-probability |
| law | MMLU law subjects, five LegalBench tasks with fixed labels, CaseHOLD | letter log-probability |
| math | GSM8K test (300) and MATH-500 (200) | greedy generation, final answer check |
| code | MBPP sanitized test (200) | greedy generation, asserts run in a subprocess with a timeout |
| science | ARC-Challenge test (500) | letter log-probability |
| reasoning | BIG-Bench Hard, 8 tasks with short answers (200) | greedy generation, exact answer |
| multilingual | Global-MMLU in 8 languages (480) | letter log-probability |
| multilingual_math | MGSM in 10 languages (250) | greedy generation, final number |
| decision | fastino/fast-decisions single-label decisions (500) | letter log-probability |

Multiple-choice skills use `wsp.mcq.score_items` and the round 6 prompt,
so their numbers line up with rounds 6 to 10. Generation skills use the
model's chat template and greedy decoding, with 768 new tokens (2048 for
reasoning models). The decision skill uses the open benchmark that the d1
model cards report. The d1 models themselves need transformers 5.x and
their own decision head, so the 4.57.1 image does not run them.

The MATH answer check is a light normalizer, not a symbolic checker. It
undercounts answers written in a different but equal form. Treat the
screens as a ranking signal across models, not as leaderboard numbers.

Rules that keep generation scores fair across chat, base and reasoning
models:

- Generation stops at the eos ids and at any end-of-turn token in the
  vocabulary (`<|eot_id|>`, `<|im_end|>`, `<|end|>`, `<end_of_turn>` and
  others), and ignores sampling settings in a checkpoint's
  generation_config.
- Three chat models ship no chat template: SaulLM-54B-Instruct, law-chat
  and Llama3-OpenBioLLM-8B. The zoo names a format for each
  (`chat_format`), and `ensure_chat_template` installs it.
- A model with no template gets the raw prompt, and its output is cut
  where it starts a new problem or turn (`cut_continuation`).
- The GSM8K and MGSM answer is the first `####` line that holds a number.
  Markdown headers such as `#### Step 1` or `#### 1. Add` do not count.
- A response that opens `<think>` and never closes it ran out of tokens
  while thinking. It has no answer and scores zero. When the template
  opens `<think>` in the prompt (the DeepSeek-R1 distills), the output gets
  that `<think>` back so the rule still applies.
- BBH answers may have markdown or a colon before them ("is: **Yes**").
  Word answers must match a whole word, so "Not" is not "No".

The code screen runs each program with `python -I` in a temp dir, in its
own process group, with an empty stdin, no captured output, and a small
environment (no tokens). A guard at the top caps memory and file size and
turns off the os and shutil calls that delete, move or kill. On a timeout
the whole process group is killed. A program passes only when it exits
with code 0 and writes a marker after the last test, so an early
`exit(0)` does not pass. The guard is for accidents, not attacks: run the
screen only in a throwaway container.

## Layer maps

`layer_skill_map(model, tokenizer, skill_texts, general_texts)` scores each
decoder layer two ways:

- Block influence: one minus the cosine between the layer's input and
  output, averaged over real tokens (ShortGPT). The map stores influence on
  skill text minus influence on general text.
- Layer drop: the increase in next-token loss when that one layer is a
  residual passthrough (its output is its input). The score is the
  increase on skill text minus the increase on general text.

The layer-drop score is the main one. A positive score means the skill
needs that layer more than general text does. Skill texts are a skill's
items with their gold answers (or reference solutions), from splits the
screens do not use. General texts are WikiText-103 validation paragraphs.

Map texts never come from the items a screen scores. For the decision
skill this holds per dataset row, because one row can hold several
decisions.

`suggest_blocks` turns the scores into contiguous blocks. Only positive
scores count as mass. The block length is the shortest span that holds 60%
of the positive mass, unless a target length is given. The map returns the
best block of that length and up to two more blocks that do not overlap
it. Blocks are `[first, last]`, both ends included, as in catalog block
records.

`block_mass` comes from the same texts that chose the blocks, so it is
optimistic. The `heldout` field checks it at no extra cost. The texts on
each side are split into even and odd positions. Blocks are chosen on the
even half only (`heldout.blocks`, `heldout.select_mass`), and their share
of the positive mass is measured on the odd half (`heldout.check_mass`),
which took no part in the choice. `heldout.split_half_r` is the
correlation of the two halves' layer scores. A low check mass or a low
correlation means the map needs more texts (`--map-texts`).

## Expert maps

`expert_skill_map` hooks the router of each MoE layer and counts how often
each expert is picked (top k) on skill text and on general text. The log
ratio of the two frequencies ranks the experts. Each layer keeps up to 8
experts with a positive ratio. The entry stores only those expert ids per
layer.

The router hook works with both transformers lines. In 4.x the router is
a `gate` module that returns logits, and the experts are a ModuleList. In
5.x the router returns `(logits, scores, indices)`, the experts are one
fused module, and `top_k` and `num_experts` sit on the router or in the
config. When the router output has an index tensor, the map uses it, so
LFM2-MoE's expert bias is counted as the model applies it. In Gemma 4 the
decoder layer holds `experts` and `router` itself, and the layer counts as
the MoE module.

## Transformers 4.x and 5.x

Layer skipping is a forward hook on the decoder layer that returns the
layer's input. Decoder layers return a tuple in 4.x and a plain tensor in
5.x, and the hook keeps whichever shape the layer returned. The influence
hooks read the output the same way. `layer_input` finds the input in
`args[0]` or in `kwargs["hidden_states"]`. Because the layer still runs and
only its output is replaced, layers that write shared state (the KV source
layers in Gemma 4) keep working when skipped.

`tests/test_atlas_tf5.py` builds tiny random models of LFM2, LFM2-MoE,
Qwen3.5, Qwen3.5-MoE and Gemma 4 (dense and MoE) from configs. For each it
checks that skipping a layer gives the same loss as deleting it, that the
layer map runs, and that expert counts sum to top-k times the token count.
A family is skipped when its config class is not in the installed
transformers, so the file passes under both 4.57.1 and 5.23.0.

## Entries

One JSON file per model:

```json
{"model": "Qwen/Qwen2.5-7B-Instruct",
 "skills": {"knowledge": 0.715},
 "skill_details": {"knowledge": {"score": 0.715, "count": 1000, "by_subject": {}}},
 "layer_maps": {"math": {"scores": [], "influence": [], "blocks": [[12, 19]], "block_mass": [0.64]}},
 "expert_maps": null,
 "params_b": 7.62, "layers": 28, "hidden": 3584, "family": "qwen2",
 "measured_at": "2026-11-14T12:00:00Z"}
```

`write_entry`, `read_entry` and `load_entries` read and write them.

## How the compiler and planner use the atlas

- `best_models_for(skill, entries)` ranks models by their screen score,
  smaller models first on ties. The planner uses it to pick the host and
  the donors for a domain.
- `best_block_for(skill, entries, base_constraints)` picks the donor block
  to graft. Constraints can fix the hidden size (so no stitch is needed),
  the model type, the allowed models, the largest donor, and the shortest
  and longest block. A block longer than the limit is cut to its best
  window of that length.
- `catalog_rows(entry, skill, units_dir)` writes blocks as catalog `block`
  records and expert maps as `expert` records with units files, in Person
  A's catalog format (`person_a/catalog/<model>.jsonl`). The compiler then
  grafts them by id, as it does mined parts.

The atlas works at model and layer level. Person A's miner works one level
down: it tests neuron clusters, heads and experts by ablation, with an
agent writing the tests. An atlas block is a good place to look for mined
parts, and a mined part is a finer cut inside an atlas block.

## Run

```bash
MODAL_PROFILE=trisanth .venv/bin/modal run modal_atlas.py --only Qwen/Qwen2.5-0.5B-Instruct --scale 0.1
MODAL_PROFILE=trisanth .venv/bin/modal run modal_atlas.py --tf5 --only Qwen/Qwen3.5-0.8B --scale 0.1
python scripts/modal_fanout.py modal_atlas.py atlas --profiles trisanth,ztrisanth
```

`--tf5` runs only the models whose zoo entry needs transformers 5.x, on a
second image that pins transformers 5.23.0 (torch 2.7.1 as before). It
takes no gated models and no giants (`--giants` is ignored with it), and it
leaves the models that need 4.x out of the skipped file. Without `--tf5`
the run is the 4.57.1 image as before. Under `--tf5` the loader tries
`AutoModelForCausalLM`, then `AutoModelForImageTextToText`, then
`AutoModel`, and loads a saved generic fast tokenizer as that class
(transformers 5 otherwise rebuilds some tokenizers and ignores
`tokenizer.json`). `--tf5` composes with `--shard`, `--nshards` and
`--only`.

Options: `--skills math,code`, `--scale` (fraction of each screen),
`--map-texts` (texts per side for the maps, default 32), `--gen-tokens`,
`--batch-size`, `--no-screens`, `--no-layer-maps`, `--gated`, and
`--giants` (H200:8, expert maps only). Each shard writes
`results/atlas_shard<i>.json`, `results/atlas_skipped_shard<i>.json` and
`results/atlas_failed_shard<i>.json` (model id and error for each cell
that failed). Every entry goes to `atlas/entries/` locally and to
`/vol/atlas/entries/` on the `wsp-models` volume. The container also saves
the entry after each skill, marked `"partial": true`, so a cell that
times out keeps the skills it finished.
