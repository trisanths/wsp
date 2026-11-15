# wsp: language and compiler reference

Sapien is a language for writing a model as a program over components mined
from existing open models, and a compiler that builds one network from it.
A request names a base (often small), the skills wanted, and limits (total
parameters, parameters run per token, heal time). Components are layer
blocks, attention heads, neuron clusters, MoE experts, and MLPs. The only
training is the stitches at the seams and a short heal (LoRA or
distillation). Running a source model at inference, output ensembles, and
hand-written weights are not Sapien.

```bash
sapien check prog.sap               # parse, validate, estimate size against the budget
sapien compile prog.sap -o out      # one model directory; wsp.load(out) returns (model, tokenizer)
sapien compile prog.sap -o ctl --control random   # random components of the same size
sapien score out --skills gsm8k,mmlu
```

Start with [docs/quickstart.md](quickstart.md) (three requests as
programs) and [docs/language.md](language.md) (every statement).

## Status

Built and tested on CPU with tiny models: the text syntax and the Python
`Program` class (graft, gate, experts, insert, lora, heal with distillation,
budget, random control on every component), one compile path and one
directory format for all of them, the `sapien` command, and the size
estimate in `sapien check`.

Shown by the logs ([results/README.md](../results/README.md)):

- The mechanics work inside one tokenizer family. A trained stitch on the
  donor's own LM loss lets a donor block compute what it computes in its own
  model (a penalty of +0.008 and -0.02 nats; run 14, Qwen donors in a Qwen
  base). Across tokenizers the stitch closes only a third to a half of the
  gap (stitch study, Llama-3.2-1B host).
- Taps, which are not Sapien, beat fine-tuning on knowledge: a Qwen2.5-7B tap
  moved Llama-3.2-1B from 47.5 to 67.9 on MMLU against 50.3 for a LoRA
  (run 6). They add nothing to a strong host (run 9).

Not proven:

- No Sapien program has beaten its controls yet. Donor blocks did not beat a
  random block or a rank-1 LoRA (run 3). A block written into a 0.5B base
  gained as much from a random block (run 14b). A skill MoE with specialist
  MLPs did not clearly beat the base or the base plus LoRA on GSM8K, MBPP,
  MMLU, or MedMCQA (differences were within the seed spread, up to about 1
  point on MedMCQA), and its router stayed on the base MLP (run 16b). Carving a model to half its depth
  collapses it, and a short heal recovers little (run 15).
- Distillation heals (run 14c), gated fusion (run 16), and inserted blocks
  against controls (run 17) were running when this file was written and are
  not in the log.
- The new unified compile paths (experts, inserts, gates, distill, budget)
  have only run on tiny random models on CPU. No large-model run has used
  them through `sapien compile`.
- The budget's `heal_minutes` stops the heal at the limit, so two arms under
  one limit can run different numbers of steps. The report gives `steps_run`;
  compare arms by it. The size estimate
  of `sapien check` is from configs and is exact only for the cases the tests
  cover (dense hosts, plain or linear-stitched experts, inserts).

This repo holds the language and compiler (part 2) and the eval harness and extraction tools (part 1b, [wsp_eval/README.md](../wsp_eval/README.md)). The team split and each
person's agent brief are in [briefs/](../briefs/). Checked facts about Deletion
Tuning are in [VERIFY.md](../VERIFY.md).

## What works now

- Written models (`wsp/carve.py`): one plain Hugging Face model at the
  requested size, from the parts of existing models. `from` a source,
  `splice` layer segments of same-shape models, `carve` layers (by block
  influence, uniform, or domain influence) and width to a parameter
  budget, then heal a LoRA and merge it. The output loads with
  `AutoModelForCausalLM.from_pretrained` (and so vLLM) with no wsp code.
  See "Writing a model from parts" in docs/language.md.
- Grafts across model families. A part from any donor model runs inside a
  host at a chosen layer: `h = h + W_out(part(W_in(h)))`. `W_out` starts at
  zero, so a fresh graft leaves the host unchanged. Part types:
  - `mlp_cluster`: a set of MLP neurons from one donor layer.
  - `attn_heads`: a set of query heads from one donor layer.
  - `expert`: one or more MoE experts from one donor layer.
  - `block`: whole donor layers i..j, run with the donor's own layer code.
  Attention and block parts keep their own KV cache. It follows the host
  cache through generate, beam search, and chunked prefill, so cached
  output equals uncached output. Static caches are not supported.
- `.lora(rank=... or budget_params=...)`: a host LoRA trained in the same
  heal loop, the fine-tuning baseline grafts must beat.
- Parts run the donor's own Transformers classes, and only the part's
  tensors are fetched from safetensors by HTTP range request.
- `W_in` can start from a ridge fit on host and donor hidden states matched
  by token end offsets. The compile report gives the fit's R squared.
- `stitch={...}` on a graft trains `W_in` as a stitch before heal
  (`wsp/stitch.py`). Kinds: linear, lowrank, mlp, multi (several host
  layers). Objectives: `mse` on matched states, `donor_lm` (the donor's own
  next-token loss, same tokenizer only), and `downstream` (match the
  donor's own states a few layers later, any tokenizers). The compile
  report records the stitch penalty or loss before and after.
- The ridge fit's default regularizer is relative 1e-6, so an exact linear
  map comes back exactly. The first position never enters the fit.
- `sink=True` (the default) on a graft: the compiler records the donor's
  mean state at position 0 at the part's input point over the align texts.
  Position 0 is the donor's attention sink (the BOS, or the first text
  token when there is no BOS). The graft feeds that state to the part at
  each sequence's first real token, also under left padding, and not again
  on cached decode steps. The stitch trains with the same sink.
  `sink=False` turns it off. Without align texts the sink stays off.
- `rms_match=True` on a graft scales each mapped vector to the donor's mean
  RMS at the input point, times a trained scalar that starts at 1.
- `part_lora_rank=r` on a graft puts a rank-r LoRA on the linears inside the
  donor part (the receiver). Heal trains it with the glue, and the compile
  report counts it as `part_lora_params`.
- A heal pass trains only the glue (and a part LoRA). `control="random"`
  swaps in a random part of the same size from the same donor layer, with
  the same options.
- Taps: `.tap(donor, donor_layer, at)` lets host layer `at` read a frozen
  donor's own layer states through a gated cross-attention branch. The
  donor is any Hugging Face model and reads the prompt with its own
  tokenizer. `.tap_train(data="general", ...)` trains the taps first on
  (prompt, completion) pairs from a data pool (`wsp/data.py`), then a LoRA or graft
  heal trains with the taps frozen. `wsp.load` reloads the donors and
  auto-prepares, so plain `model.generate(input_ids, attention_mask=...)`
  works, also in the eval harness. `wsp.tap_cost(model)` estimates the
  decode and donor prefill cost. See docs/language.md.
- Mask algebra and task vectors on the host, as before.
- `wsp.load(path)` returns `(model, tokenizer)` for any compiled directory.
- Stub extractors: `top_activation_neurons` and `top_routed_experts`.

Not built yet: vLLM serving for grafted models, and a deletion budget rule.

## Example

```python
from wsp import Catalog, Program, load

catalog = Catalog.load("catalog.jsonl")
program = (
    Program(host="unsloth/Llama-3.2-1B-Instruct")
    .graft("qmath_l18_top512")
    .heal(steps=300, data="general")
)
program.compile("outputs/math_graft", catalog, align_texts=texts)
model, tokenizer = load("outputs/math_graft")
```

Mask algebra on the host still works:

```python
from wsp import use

program = Program(
    host="unsloth/Llama-3.2-1B-Instruct",
    masks=(use("mask_mmlupro_biology_llama1b") | use("mask_mmlupro_chemistry_llama1b")),
)
```

Agents can send the same program as JSON: `program.to_dict()` and
`Program.from_dict(data)`.

## Base model

The catalog uses `unsloth/Llama-3.2-1B-Instruct`, an ungated copy of
`meta-llama/Llama-3.2-1B-Instruct`. The meta repo needs a Hugging Face access
token. All components in one program must name the program's base; the
compiler refuses a mixed-base program.

## Run

wsp needs transformers 5.23.0 (the `compile` extra and the Modal images
pin it). Expert parts read the fused 3D expert tensors of Transformers 5
MoE classes, and LFM2, LFM2 MoE, and LFM3 donors work.

CPU tests (toy tensors only, no model downloads):

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]" && .venv/bin/pytest -q
```

Real models run on Modal, profile `trisanth`:

```bash
MODAL_PROFILE=trisanth .venv/bin/modal run modal_app.py::verify
```

```bash
MODAL_PROFILE=trisanth .venv/bin/modal run modal_app.py::compile_json --program-path program.json --name my_model
```

Compiled models go to the Modal volume `wsp-models` at `/vol/compiled/<name>`.

## Mask format

Masks use the Deletion Tuning `packbits_zstd` format (code from
agentic-learning-ai-lab/deleting-weights, MIT license). The file stores KEEP
bits. In this code, a mask is a dict of parameter name to a boolean tensor
where True means delete.
