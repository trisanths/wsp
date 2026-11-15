# The wsp text syntax

A wsp program names a host model and the parts to add to it. The text form
is for people. It parses into the same `wsp.program.Program` object that
agents send as JSON, and it prints back to the same text.

```wsp
host unsloth/Llama-3.2-1B-Instruct
seed 0
mask (bio | chem) - law restrict layers 4..12 modules mlp
tv code_tv weight 0.5 layers 10..20 modules mlp
lora rank 16 targets q_proj,v_proj
graft qmath_l18_contrast512 at 6 rank 64 control random seed 101
graft qmath_block_8_19 at 5 rank 64 stitch mlp objective downstream steps 500
heal steps 300 data gsm8k_train lr 3e-4
```

A tap program (in run 6, a tap from Qwen2.5-7B-Instruct gained 20.4
points on MMLU over the host, and a rank-16 LoRA gained 2.8):

```wsp
host unsloth/Llama-3.2-1B-Instruct
tap Qwen/Qwen2.5-7B-Instruct layer 20 at 10
tap_train data mmlu_aux steps 500 batch_size 8 seq_len 256 branch_lr 3e-4 gate_lr 3e-2
```

The code is in `wsp/lang.py`:

```python
from wsp.lang import parse, load_program, format_program

program = load_program("math.wsp")      # or parse(text)
text = format_program(program)          # parse(text).to_dict() == program.to_dict()
```

## Rules

- One statement per line. The first word names the statement.
- `#` starts a comment that runs to the end of the line.
- Options follow the statement as `name value` pairs, in any order.
  `stitch` is the one exception: it reads to the end of the line.
- An id (catalog id, model name, local path, module name, data name) uses
  letters, digits, and `_ - . /`. It cannot start with `-`. So a local
  model folder such as `/models/tiny` works as a host.
- Layer lists use inclusive ranges and commas: `4..12`, `1,3,5..7`.
- Name lists use commas: `q_proj,v_proj`. A space after a comma is fine.
- Numbers keep their type: `1` is an integer and `1.0` or `3e-4` is a float.
- Statements can come in any order. The order of `tv`, `graft`, and `tap`
  lines is kept. `host`, `seed`, `mask`, `lora`, `tap_train`, and `heal`
  can each appear once.
- An error names the line, for example `line 3: at needs an integer, got 'x'`.

## Statements

### host

The model that the program changes. `base` is an older name for it.
Every program needs one.

```wsp
host unsloth/Llama-3.2-1B-Instruct
```

### seed

The random seed for the compile and the heal pass. The default is 0.

```wsp
seed 3
```

### mask

A deletion mask from catalog masks. The operators work on deletions:

| Operator | Deletes a weight when |
|---|---|
| `a \| b` | a or b deletes it |
| `a & b` | a and b both delete it |
| `a - b` | a deletes it and b does not |

`-` binds tightest, then `&`, then `|`, the same as in Python. Operators
of one kind group to the left, so `a - b - c` is `(a - b) - c`. Put spaces
around `-`, because an id can hold a `-`.

`restrict layers L modules M` keeps deletions only in those layers and
modules. Either option can be left out. At the end of the line, `restrict`
applies to the whole expression. Inside parentheses, it applies to the
part in the parentheses.

```wsp
mask (bio | chem) - law restrict layers 4..12 modules mlp
```

```wsp
mask (bio restrict layers 0..7) | (chem restrict modules self_attn)
```

### tv

A task vector: add `weight * (fine-tuned - base)` on the chosen layers and
modules. `weight` defaults to 1.0. Without `layers` or `modules`, it covers
all of them. A program can have many `tv` lines.

```wsp
tv code_tv weight 0.5 layers 10..20 modules mlp
```

### lora

A LoRA on the host decoder layers, trained by the heal pass. Give `rank`,
or `budget_params` (also written `budget`) to pick the largest rank whose
parameter count fits. `targets` defaults to the compiler's list, `alpha`
to 2 times the rank, and `layers` to all layers. A LoRA needs a `heal` line.

```wsp
lora rank 16 targets q_proj,v_proj alpha 32 layers 0..7
heal
```

```wsp
lora budget_params 460000
heal steps 300
```

### graft

Graft a catalog part into the host. A program can have many `graft` lines.

| Option | JSON key | Meaning |
|---|---|---|
| `at N` | `at` | host layer; left out, the compiler picks the matching depth |
| `control random` | `control` | swap in random units of the same size from the same donor layer |
| `seed N` | `control_seed` | seed for the random control (default 0) |
| `rank N` | `glue_rank` | rank of the glue; left out, the glue is full rank |
| `sink off` | `sink` | turn off the attention sink feed (on by default) |
| `rms_match on` | `rms_match` | scale mapped vectors to the donor's mean RMS (off by default) |
| `part_lora_rank N` | `part_lora_rank` | a rank-N LoRA inside the donor part, trained by heal |
| `stitch ...` | `stitch` | train `W_in` as a stitch before heal (see below) |

`control_seed` and `glue_rank` also work as option names.

```wsp
graft qmath_l18_contrast512 at 6 rank 64
graft qmath_l18_contrast512 at 6 rank 64 control random seed 101
graft qmath_l18_top512 sink off rms_match on part_lora_rank 4
```

`stitch` goes last on the line. Its first word can be the kind (`linear`,
`lowrank`, `mlp`, or `multi`). Then come `name value` pairs that go into
the stitch dict as they are: `objective`, `steps`, `lr`, `batch_size`,
`seq_len`, `seed`, `rank`, `hidden`, `offsets`, `end_layer`. Values can be
numbers, names, `true`, `false`, `none`, or a list in brackets. A `rank` or
`seed` after `stitch` belongs to the stitch, not to the graft.

```wsp
graft qmath_block_8_19 at 5 rank 64 stitch mlp objective downstream steps 500
graft qmath_l18_top512 stitch multi offsets [-2, 0] objective mse lr 1e-3
graft qmath_l18_top512 stitch lowrank rank 32
```

### tap

A tap: the donor reads the prompt with its own tokenizer and runs up to
donor layer `layer`. At host layer `at`, a small gated cross-attention
branch lets each host token read those donor states. The gate starts at
0, so an untrained tap leaves the host unchanged. The donor is a Hugging
Face id or a local model folder, from any model family. A program can
have many `tap` lines.

| Option | JSON key | Meaning |
|---|---|---|
| `layer N` | `donor_layer` | donor decoder layer the tap reads (required) |
| `at N` | `at` | host decoder layer the branch writes (required) |
| `dim N` | `attn_dim` | width of the branch attention (default 256) |
| `heads N` | `heads` | attention heads; `dim` must be a multiple (default 8) |
| `id NAME` | `tap_id` | a name for the tap in reports (default `tap0`, `tap1`, ...) |

`donor_layer`, `attn_dim`, and `tap_id` also work as option names. Taps
that read the same donor share one donor run per prompt. Taps that read
the same donor and layer also share one memory.

```wsp
tap Qwen/Qwen2.5-7B-Instruct layer 20 at 10
tap Qwen/Qwen2.5-7B-Instruct layer 27 at 12 dim 128 heads 4 id late
tap LiquidAI/LFM2-350M-Math layer 8 at 8
```

### tap_train

How the taps train. A program with `tap` lines needs a `tap_train` line,
and a `tap_train` line needs a tap. Training uses (prompt, completion)
pairs from `wsp/pairs.py`: `gsm8k_train`, `mmlu_aux`, or `trivia_train`.
The donors read only the prompt, and the loss covers only the
completion. The gates train at `gate_lr`, and the rest of each branch at
`branch_lr`. `tap_train` alone uses the round 6 MMLU settings:
`data mmlu_aux steps 500 batch_size 8 seq_len 256 branch_lr 0.0003 gate_lr 0.03`.

```wsp
tap Qwen/Qwen2.5-7B-Instruct layer 20 at 10
tap_train data trivia_train steps 500 batch_size 8 seq_len 256 branch_lr 3e-4 gate_lr 3e-2
```

The taps train first. When the program also has a `lora` or grafts and a
`heal` line, heal runs second with the taps frozen, on the pairs named by
the heal `data`. Run 6 trained a tap and a LoRA together, and the LoRA
took over while the gate stayed near 0, so the compiler always stages
them.

```wsp
tap Qwen/Qwen2.5-7B-Instruct layer 20 at 10
tap_train data mmlu_aux
lora rank 16
heal steps 500 data mmlu_aux lr 1e-4 batch_size 8 seq_len 256
```

### heal

The heal pass that trains the glue (and the LoRA). `heal` alone uses the
defaults: `steps 300 data gsm8k_train lr 0.0003 batch_size 8 seq_len 384`.
In a program with taps, heal trains after the taps (see `tap_train`), and
its `data` names a pairs set in `wsp/pairs.py`.

```wsp
heal steps 300 data gsm8k_train lr 3e-4 batch_size 8 seq_len 384
```

## Writing a model from parts

A written program makes one standard model at the size you ask for, from
the parts of existing models. No donor runs beside it at inference, it is
never larger than the target, and the output directory is a plain Hugging
Face model: `AutoModelForCausalLM.from_pretrained` and vLLM load it with
no wsp code. The code is in `wsp/carve.py`.

```wsp
from Qwen/Qwen2.5-14B-Instruct
carve target 7.6b select influence
lora rank 64
heal steps 1000 data mmlu_aux lr 1e-4 batch_size 8 seq_len 512
```

The compile runs these steps in order:

1. Load the source model, or every splice source (bf16).
2. `splice`: build one model from layer segments of the sources.
3. `carve`: choose and keep layers, then cut the width of every layer.
4. `lora` and `heal`: train a LoRA on the heal data, then merge it into
   the weights.
5. Save the model, its tokenizer, `program.json`, and
   `compile_report.json`.

A written program cannot have masks, task vectors, grafts, or taps. In a
written program, `heal` needs a `lora`, since the LoRA is what it trains.

### from

The source model. The program's host is this model. A written program
needs a `from` line or a `splice` line.

```wsp
from Qwen/Qwen2.5-32B-Instruct
carve keep 27 select uniform
```

### carve

Cut the model down. Give `keep N` (the layer count) or `target S` (a size
in billions of parameters, written `7.6b` or `7.6`), not both. With a
target, the compiler keeps the most layers that fit, after the width cut.
It counts the largest layers first, so the result fits whichever layers
are kept. Embeddings (tied or not) and norms count toward the size.

| Option | JSON key | Meaning |
|---|---|---|
| `keep N` | `keep` | keep N decoder layers (2 or more) |
| `target 7.6b` | `target_params_b` | keep the most layers that fit 7.6 billion parameters |
| `select R` | `select` | `influence` (default), `uniform`, or `domain` |
| `domain NAME` | `domain_data` | a pairs name (`wsp/pairs.py`) for the domain texts of `select domain` |
| `weight W` | `domain_weight` | the domain weight for `select domain` (default 0.75) |
| `width mlp F heads F` | `width` | keep the fraction F of MLP neurons and of query heads in every layer |

Selection rules:

- `influence`: score each layer by block influence, the mean of
  1 - cosine(layer input, layer output) over the calibration tokens
  (ShortGPT, 2024). Keep the first layer, the last layer, and the
  highest-scoring layers between them.
- `uniform`: keep evenly spaced layers from the first to the last.
- `domain`: score each layer on the domain texts and on the general texts,
  and rank by `weight * domain + (1 - weight) * general`. The first and
  last layers stay.

The calibration (general) texts are the `calib_texts` passed to the
compile, or else the first 256 heal texts. The domain texts are the
`domain_texts` passed to the compile, or else the `domain` pairs. The
command line has no option for `calib_texts`, so there a carve without a
`heal` needs `select uniform`.

`width` slices the same fraction of MLP neurons and of query heads out of
every layer, so the config stays plain (one `intermediate_size`, one head
count). Grouped-query attention stays consistent: every kept KV head
serves the same number of query heads (see `wsp/prune.py`). The units with
the highest mean activation on the calibration texts stay. Without
calibration texts, a weight-norm score picks them.

```wsp
from Qwen/Qwen2.5-14B-Instruct
carve target 7.6b select domain domain gsm8k_train weight 0.9 width mlp 0.75 heads 0.75
lora rank 32
heal steps 500 data gsm8k_train
```

### splice

Build one model from layer segments of several models with the same
architecture and hidden size, for example the 7B Qwen2.5 Instruct, Math,
and Coder models. Every planned layer must have the same parameter shapes
as the base layers, or the compile stops with an error. The compile also
stops when a source differs from the base in a config field that a layer
copies when it is built (the head counts, `head_dim`, `hidden_act`,
`rms_norm_eps`, `sliding_window`, and the like). The saved model rebuilds
each layer from the base config, so such a layer would compute something
else after the save.

| Option | Meaning |
|---|---|
| `base KEY` | the source whose embeddings, final norm, head, tokenizer, and config the model keeps |
| `sources KEY=MODEL,...` | the source models; a key uses letters, digits, and `_` |
| `plan KEY:A..B, KEY:C` | the layers of the new model, in order; ranges include both ends |

```wsp
splice base q sources q=Qwen/Qwen2.5-7B-Instruct,m=Qwen/Qwen2.5-Math-7B-Instruct,c=Qwen/Qwen2.5-Coder-7B-Instruct plan q:0..9, m:10..18, c:19..27
lora rank 16
heal steps 300 data gsm8k_train
```

The host of a splice program is the base model. Every layer runs with the
base config (for example its `rope_theta`), so the compile report lists
the config fields where a source differs from the base. A splice and a
carve can go in one program: the carve then works on the spliced model.

### The compile report

`compile_report.json` of a written program has `written: true`, the
parameter count and layer count before and after, `splice` (the plan and
the config differences), `carve` (the kept layers, the influence scores,
and the width counts), `lora`, `heal` (the per-step losses), and
`seconds` for each step. `wsp.load(path)` returns the plain model.

In Python:

```python
from wsp import Program

program = (
    Program(base_source="Qwen/Qwen2.5-14B-Instruct")
    .carve(target_params_b=7.6, select="influence")
    .lora(rank=64)
    .heal(steps=1000, data="mmlu_aux", lr=1e-4)
)
program.compile("outputs/q14_to_7b", None, calib_texts=texts)
```

The tools also work alone:

```python
from wsp.carve import carve_layers, carve_width, choose_layers, layer_influence, layers_for_budget, splice

scores = layer_influence(model, tokenizer, texts)
carve_layers(model, choose_layers(scores, layers_for_budget(model, 7.6e9), "influence"))
carve_width(model, keep_mlp=0.75, keep_heads=0.75)
model.save_pretrained(out_dir)
```

## Text to JSON

`parse(text).to_dict()` gives the JSON that agents send:

| Text | JSON |
|---|---|
| `host X` | `"host": "X"` |
| `seed 0` | `"seed": 0` |
| `mask a \| b` | `"masks": {"op": "union", "a": {"op": "use", "id": "a"}, "b": {...}}` |
| `mask a & b`, `mask a - b` | `"op": "intersect"`, `"op": "difference"` |
| `... restrict layers 4..6 modules mlp` | `{"op": "restrict", "inner": {...}, "layers": [4, 5, 6], "modules": ["mlp"]}` |
| `tv t weight 0.5` | `"task_vectors": [{"id": "t", "weight": 0.5, "layers": null, "modules": null}]` |
| `lora rank 16 targets q_proj,v_proj` | `"lora": {"rank": 16, "budget_params": null, "targets": ["q_proj", "v_proj"], "alpha": null, "layers": null}` |
| `graft g at 5 rank 64` | `"grafts": [{"id": "g", "at": 5, "control": null, "control_seed": 0, "glue_rank": 64}]` |
| `... stitch mlp steps 500` | `"stitch": {"kind": "mlp", "steps": 500}` in that graft |
| `... sink off`, `rms_match on`, `part_lora_rank 4` | `"sink": false`, `"rms_match": true`, `"part_lora_rank": 4` |
| `heal steps 300` | `"heal": {"steps": 300, "data": "gsm8k_train", "lr": 0.0003, "batch_size": 8, "seq_len": 384}` |
| `tap d layer 20 at 10` | `"taps": [{"donor": "d", "donor_layer": 20, "at": 10, "attn_dim": 256, "heads": 8, "tap_id": null}]` |
| `tap_train steps 500` | `"tap_train": {"data": "mmlu_aux", "steps": 500, "batch_size": 8, "seq_len": 256, "branch_lr": 0.0003, "gate_lr": 0.03}` |
| `from X` | `"base_source": {"model": "X"}`, and `"host": "X"` |
| `carve keep 22 select influence` | `"carve": {"keep": 22, "target_params_b": null, "select": "influence", "domain_data": null, "width": null}` |
| `carve target 7.6b ... width mlp 0.75 heads 0.75` | `"target_params_b": 7.6`, `"width": {"keep_mlp": 0.75, "keep_heads": 0.75}` |
| `splice base q sources q=X,m=Y plan q:0..1, m:2` | `"splice": {"sources": {"q": "X", "m": "Y"}, "base": "q", "plan": [["q", 0], ["q", 1], ["m", 2]]}` |

A missing statement gives its JSON default: `"masks": null`, empty
`task_vectors` and `grafts`, `"heal": null`, and no `lora`, `taps`,
`tap_train`, `base_source`, `carve`, or `splice` key. So a program
without these keys has the same JSON (and the same hash) as before they
existed. `format_program` writes `seed` and every
`heal` and `tap_train` field, and leaves out graft, tap, and tv options
that hold their default.

## Command line

```bash
.venv/bin/python -m wsp.lang check math.wsp      # parse and print the JSON
.venv/bin/python -m wsp.lang format program.json  # print a JSON program as text
.venv/bin/python -m wsp.lang compile math.wsp outputs/math --catalog catalog.jsonl
```

A written program needs no `--catalog`.

`check` exits with 1 and prints `file: line N: ...` on an error. A `.json`
file with a key that `wsp.lang` does not know is an error too, so no
option is lost on the way to text. `compile`
takes `--align N` (heal texts for the `W_in` fit when the program has
grafts, default 200, 0 skips the fit) and `--device`.

## How compile works

1. `wsp.lang` parses the file into a `Program`. The compiler checks that
   every mask and task vector in the catalog is built on the program's
   host. Grafts can come from any model.
2. The compiler sets the seed, then loads the host and its tokenizer.
3. Task vectors go onto the host weights in order, then the mask deletes
   weights.
4. The LoRA, if any, wraps the chosen host linears.
5. Each graft fetches its part's tensors from the donor, fits `W_in` on the
   align texts, trains the stitch if one is set, and attaches at its layer.
6. Taps: the compiler loads each donor once (bf16), attaches the
   branches, and trains only the taps on the `tap_train` pairs.
7. The heal pass trains only the glue, part LoRAs, and host LoRA. With
   taps, the taps are frozen first, and heal trains on pairs while the
   taps read the donors. The output directory gets the model,
   `program.json`, `program.wsp`, and `compile_report.json`.
   `wsp.load(path)` reads it back.

A tap program also saves `taps.json` (branch shapes, layers, final gates,
and donor ids) and `tap_weights.safetensors`, plus `grafts.json` (an empty
list when it has no grafts, so the eval harness loads it with
`wsp.load`). `compile_report.json` has a `taps` entry: the spec, the
per-step losses, the final gates, the trainable parameter count, the
seconds, and `cost` from `wsp.tap_cost` (rough FLOPs per decoded token
and donor prefill FLOPs per prompt, for 128 donor tokens).

`wsp.load` loads each donor by its id on the same device and turns on
auto-prepare, so plain `model(input_ids)` and
`model.generate(input_ids, attention_mask=mask)` work. On a prefill call,
the taps decode each row's host ids back to text, run the donors on those
texts, and match the visibility to the given ids. When the ids do not
encode back the same, a host token reads only donor tokens that end at
or before its own end in the decoded text. `generate` builds the context
once from its whole `input_ids` before the first forward call, so
chunked prefill, `use_cache=False`, and beams all read the same context,
and the donors run once per prompt. Outside `generate`, every call with
an empty cache is a new prompt, and calls with a cache reuse the
context. A manual `wsp.tap.prepare` takes priority until
`wsp.tap.clear_tap_context`. The whole first input counts as the
prompt: to score a completion as training does, prefill the prompt
alone and then feed the completion with the cache from that prefill.
Host special tokens (chat template markers, BOS) add no text, so the
donors read the prompt without them; `prepare` passes the prompt string
as it is, so for a chat-template prompt the two differ.
