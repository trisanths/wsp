# Mixed-family compiler: design

Date: 2026-11-14. Owner: Trisanth (part 2). Status: draft for approval.

## Intent

The project goal is to find the best combinations of model parts without
regard to model family. Parts come from many open-weight models. A person or
an agent writes a program that names parts. `compile()` joins them into one
working model. Same-base mask algebra alone only trims weights from one model,
so it is not the goal. It stays as one way to carve a part out of a donor.

The output is one fixed combined model per task type. Routing at inference
time is out of scope.

## Goals

- Parts as fine as one MLP neuron cluster or a set of attention heads, from any
  donor model, in any host model.
- `compile()` sizes and initializes all the glue: projections between hidden
  sizes, the skip connection around each part, and the depth where each part
  goes.
- A short heal pass that trains only the glue.
- A measured answer to one question: does a grafted part beat the same heal
  budget spent with no graft?

## Non-goals

- Inference-time routing.
- vLLM support for grafted models in the hackathon.
- Training parts from scratch. Parts come from existing weights.

## Anti-goals

- A gain that comes only from the heal training, not from the part. Every
  result reports a control with the same heal steps and data and no graft
  (glue only, or a LoRA with the same trainable parameter count).
- Search-split numbers reported as results.

## Constraints

- Hackathon timeline. The hour-8 gate is a compiled model scored by real eval.
- GPU on Modal: $250 per person (about $1,000 for the team), more on request.
  That covers a few hours of an 8-GPU node for running frontier donors during
  extraction. Local Mac runs CPU tests on toy models only.
- Host and donor tokenizers differ. A donor part never sees its own tokenizer.

## Approaches considered

1. **Grafted parts on the host residual stream (chosen).** Each part reads the
   host's hidden state through a projection, runs, and writes back through a
   projection on a skip connection. Granularity is set by the part type, from a
   layer range down to a neuron cluster. This is model stitching, the method
   with the most published evidence.
2. **Experts plus a learned router.** Each cluster becomes an expert behind a
   router. Rejected: routing is not needed, and a weak router hides whether the
   parts are good.
3. **Output-level ensembles** (run whole models, combine logits). Rejected:
   tokenizers differ, cost is a full model per part, and it is not composition
   of parts.

The user chose approach 1 with the finest granularity that is a good tradeoff.

## Design

### Part types

A part is a function on a donor's residual stream at one depth. It maps a
donor hidden vector to a donor-sized update.

| Type | What it holds | Params (donor size d, n units) |
|---|---|---|
| `mlp_cluster` | Neurons S of the MLP in donor layer l: rows S of gate_proj and up_proj, columns S of down_proj, and that layer's post-attention RMSNorm | about 3 d n |
| `attn_heads` | Query heads H of donor layer l, with the KV heads they use (GQA), the output-projection columns for H, the input RMSNorm, and the donor's rotary settings | about 2 d n h + KV |
| `expert` | One MoE expert's MLP in donor layer l | about 3 d n_expert |
| `block` | Whole donor layers i to j, run in order | full layers |

A deletion mask can apply to a donor before a part is cut from it. That keeps
the same-base mask algebra as a carving tool.

`mlp_cluster` and `attn_heads` are the fine grain. `block` is the coarse
fallback when fine parts do not hold their function outside their model.

### Any family, any size (user decision, 2026-11-14)

The compiler takes parts from any open-weight model, including the largest
frontier MoE models. Two rules make that possible without per-family code or
whole-model loads:

- **Load only the part's tensors.** Safetensors shards have a JSON header with
  each tensor's byte offsets. The compiler reads the header, then fetches only
  the tensors a part needs with HTTP range requests. A single expert or one
  layer from a model of hundreds of billions of parameters is a small download.
- **Run the donor's own code.** The compiler builds the donor's own layer or
  MLP module from its Hugging Face config (with `trust_remote_code` where the
  model needs it) and loads the fetched tensors into it. Attention variants,
  norm styles, and biases then come from the donor's code, not ours. A neuron
  cluster is a sliced copy of that module.

MoE experts are a first-class part type, `expert`: one expert's MLP from one
layer. They are the natural unit in most frontier open models.

What still costs real compute is anything that runs the whole donor:
extraction (finding which clusters or experts do what) and the ridge fit for
`W_in`, which needs donor hidden states. When the donor is too large to run,
`W_in` starts random and the heal pass trains it, and the compile report marks
the fit as skipped.

### Graft

At host layer k, after the host layer's output:

```
x = W_in @ h            # host size -> donor size
u = part(x)             # the part, with its own norm
h = h + W_out @ u       # donor size -> host size, skip connection
```

- `W_out` starts at zero, so a freshly compiled model equals the host exactly.
- `W_in` starts from a least-squares fit (next section), or random if no fit is
  possible.
- `attn_heads` parts use the host's token positions for rotary embeddings.

### Initializing W_in across tokenizers

1. Run host and donor on the same text sample (a few thousand tokens from the
   task data).
2. Match positions where a host token and a donor token end at the same
   character offset. BPE tokenizers share most word ends.
3. Fit ridge regression from host hidden state at layer k to donor hidden state
   at layer l on the matched positions. That is `W_in`.
4. Report the fit's R squared in the compile report. A low value is a warning
   that the depth pairing is poor.

### Placement

Default host depth: `k = round(l / L_donor * L_host)`. A program can set `at=`.
The R squared from the fit lets an agent compare depths cheaply before any heal.

### Heal

- Trainable: every `W_in` and `W_out`. Optional: a small scale per graft.
- Frozen: the host and every part.
- Loss: next-token loss on the task's training data in the host tokenizer.
- Default: 300 steps, AdamW, bf16, on one Modal GPU.
- Every heal reports the no-graft control with the same steps and data.

### Language

```python
from wsp import Program, part

math = part("Qwen/Qwen2.5-Math-1.5B", layer=14, kind="mlp_cluster", neurons="clusters/qmath_l14_c3.pt")
code = part("Qwen/Qwen2.5-Coder-1.5B", layers=(10, 12), kind="block")

program = (
    Program(host="unsloth/Llama-3.2-1B-Instruct")
    .graft(math)
    .graft(code, at=9)
    .heal(steps=300, data="gsm8k_train")
)
program.compile("outputs/math_code", catalog)
```

Programs keep the JSON form from today (`to_dict`, `from_dict`), so agents
send the same structure. Catalog records gain the types `mlp_cluster`,
`attn_heads`, and `block`, with fields `model`, `layer` or `layers`, and
`units` (a path to an index file).

### Compiled output

A directory with: the host model id, each part's weights (only the cut
tensors), the glue weights, `program.json`, and `compile_report.json` (fit R
squared per graft, parameter counts, heal loss curve). `wsp.load(path)` builds a
Hugging Face host model with graft hooks. Generation uses Hugging Face
`generate`.

### Effect on the team

- Person B: the eval harness must accept a `wsp.load` model and use Hugging
  Face generation for grafted models. This is slower than vLLM; keep the search
  splits small.
- Person A and Person B: "extraction" means finding neuron clusters and head
  sets with a known function in donor models, and writing them as catalog
  records with index files.
- Arjun: the action space becomes graft lists, not mask expressions.

## Testing

- CPU, toy models: two random tiny Llama-style models with different hidden
  sizes and head counts. Checks: a compiled model with zero `W_out` gives the
  host's exact logits; gradients reach `W_in` and `W_out` after one step; the
  offset matcher pairs the right positions on a hand-made example.
- Modal, first real run: host Llama-3.2-1B-Instruct, donor Qwen2.5-Math-1.5B,
  task GSM8K. Compare host, host plus no-graft heal control, and host plus
  graft plus heal, on the search split.

## Calls made after the builder check

These are design calls by Claude, not decisions the user made. Each one can be
changed.

- **Control.** The required control is the same glue wrapped around a random
  cluster of the same shape, from the same donor layer, with the same heal
  steps and data. It isolates the donor's content: the glue alone (about 6M
  parameters for a 2048 to 1536 pair) is a large adapter. A LoRA on the host,
  matched to glue plus part parameters, is the second control.
- **Splits.** Held-out: the full GSM8K test set, run once at the end. Search:
  200 items carved from GSM8K train. Heal data: the rest of GSM8K train. The
  three sets do not overlap.
- **Hour-8 gate.** The pipeline runs end to end: compile, heal, and a real eval
  score for host, control, and graft. Beating the control is the claim, not the
  gate. Claims use 3 seeds.
- **Index files.** A `.pt` file holding a dict: `kind`, `model`, `layer`, and
  `indices` (a LongTensor). For `attn_heads` the indices are query heads; the
  compiler derives the KV heads from the donor's GQA config. Until extraction
  delivers files, the compiler ships a stub extractor: the top n neurons by mean
  absolute activation on task text.
- **Generation and cache.** `mlp_cluster` acts on each token alone, so it works
  with the host's KV cache and batched `generate`. Build it first.
  `attn_heads` and `block` parts recompute their attention over the full
  sequence each step (no cache) in the hackathon build.
- **One program type.** `Program(host=...)` replaces `base=`; `base` stays as an
  alias, and old `program.json` files still load. One program can hold host
  masks, task vectors, and grafts. Order: task vectors, then masks, then grafts.
- **Fit taps.** For `mlp_cluster` the ridge target is the donor's residual after
  attention in layer l, which is the cluster's real input. For `attn_heads` and
  `block` it is the input of layer l (for `block`, the first layer i). Default
  placement for `block` uses i.
- **Masked donors.** Carving a part from a masked donor is out of scope for this
  build.

## Left to the builder

- How clusters are found in donors (Person A and Person B own extraction).
  The compiler only needs index files.
- Ridge strength, the size of the alignment sample, and the heal optimizer
  settings.
- How to cache donor weights on the Modal volume.

## Open questions

- Which tasks the hackathon tracks score. This sets the first host, donor, and
  data.
