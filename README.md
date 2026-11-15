# wsp: weight-space programming

Write a model as a program over components, then compile it into one model.

This repo holds the language and compiler (part 2) and the eval harness and extraction tools (part 1b, [wsp_eval/README.md](wsp_eval/README.md)). The team split and each
person's agent brief are in [briefs/](briefs/). Checked facts about Deletion
Tuning are in [VERIFY.md](VERIFY.md).

## What works now

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
  tokenizer. `.tap_train(data="mmlu_aux", ...)` trains the taps first on
  (prompt, completion) pairs from `wsp/pairs.py`, then a LoRA or graft
  heal trains with the taps frozen. `wsp.load` reloads the donors and
  auto-prepares, so plain `model.generate(input_ids, attention_mask=...)`
  works, also in the eval harness. `wsp.tap_cost(model)` estimates the
  decode and donor prefill cost. See docs/language.md.
- Mask algebra and task vectors on the host, as before.
- `wsp.load(path)` returns `(model, tokenizer)` for any compiled directory.
- Stub extractors: `top_activation_neurons` and `top_routed_experts`.

Not built yet: the text syntax, vLLM serving for grafted models, and a
deletion budget rule.

## Example

```python
from wsp import Catalog, Program, load

catalog = Catalog.load("catalog.jsonl")
program = (
    Program(host="unsloth/Llama-3.2-1B-Instruct")
    .graft("qmath_l18_top512")
    .heal(steps=300, data="gsm8k_train")
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
