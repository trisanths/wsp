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

A missing statement gives its JSON default: `"masks": null`, empty
`task_vectors` and `grafts`, `"heal": null`, and no `lora`, `taps`, or
`tap_train` key. So a program without taps has the same JSON (and the same
hash) as before taps existed. `format_program` writes `seed` and every
`heal` and `tap_train` field, and leaves out graft, tap, and tv options
that hold their default.

## Command line

```bash
.venv/bin/python -m wsp.lang check math.wsp      # parse and print the JSON
.venv/bin/python -m wsp.lang format program.json  # print a JSON program as text
.venv/bin/python -m wsp.lang compile math.wsp outputs/math --catalog catalog.jsonl
```

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
