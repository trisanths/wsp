# Skill deltas

Code: `wsp/delta.py`. Tests: `tests/test_delta.py`. Round: `modal_round16d.py`.

## What a skill delta is

Many specialist models are fine-tunes of one base: Qwen2.5 Math and Coder
at 1.5B and 7B, DeepSeek-R1-Distill-Qwen, Llama-3 medical and math
fine-tunes. For each weight matrix they share with the host, a specialist
differs by D = W_specialist - W_host. A skill delta keeps D as a rank-r
factor pair (A: r x in, B: out x r) and keeps the norms and biases as exact
small vectors.

Run 16b showed why the weights must run together. Qwen2.5-Math-1.5B-Instruct
scores 81.7 on GSM8K and the base 68.7, but a per-token router over MLP-only
experts stayed at the base. A delta changes attention too, and it keeps the
specialist's norms, biases, and rope base.

One network holds the host and a bank of named deltas. One expert is active
per sequence. Out = W x + scale * B (A x) for the active expert. Expert
"base" means no delta.

- Params run per token: host plus the active delta's factors.
- Stored params: host plus every delta plus the router.

## How a delta is mined

`mine_delta(host, specialist, rank)` reads both models one tensor at a time.
Each argument is a model, a local directory, or a Hugging Face id (read
lazily from safetensors). For each shared 2-D weight of the kinds q, k, v,
o, gate, up, down (and embeddings and head when `include_embeddings=True`)
it computes D in float32 on the given device and takes the SVD. 1-D
parameters (norms, biases) that differ are stored as the specialist's exact
values. Parameters missing from either model or with another shape are
listed in `attrs["skipped"]`.

`rank` is one of:

- an int, the same rank for every matrix kind
- `"full"`, which stores D itself (a reference)
- a float in (0, 1], an energy target per matrix
- a dict by kind (`q`, `k`, `v`, `o`, `gate`, `up`, `down`, `embed`, `head`, `default`)

Per matrix the delta records the relative error ||D - BA|| / ||D||, the
energy kept, and the ranks needed for 90% and 99% of the energy. That
answers how low-rank real fine-tune deltas are. `summarize_stats` gives
mean and median per kind. A delta mined at rank 256 can be cut to a lower
rank with `Delta.truncate(r)`, since the factors come from the SVD in order.

Non-weight differences are recorded in `attrs["differences"]`: rope_theta,
max_position_embeddings, chat template, generation config, and a few other
config keys.

`mine_delta(..., control="random")` and `random_control(delta, rank)` give
factors with the same shapes and the same Frobenius norm of the product per
matrix, in random seeded directions. The control carries no vectors and no
rope change. It tests whether the directions matter, not the size.

## The network

`SkillDeltaNetwork(host_model, deltas)` changes the host model in place.

- A forward hook on each adapted linear layer adds the delta. The module
  names stay the same as in the host.
- Norms and biases are copied in when the expert changes.
- Embeddings and the head, if mined, are written as dense weights when the
  expert changes. The network keeps one copy of the host table for this.
- The rope base switches with the expert when it differs from the host.
  This is the specialist's positional base. Qwen2.5-Math has rope_theta
  10000 and 4096 positions; the host and Coder have 1e6. The specialist was
  trained at its own base, so the delta runs best at it. It does mean the
  network's long-context behavior under the Math expert is the specialist's,
  not the host's.
- `merge(network, expert)` returns a plain model with one delta folded in.
  With rank `"full"` it equals the specialist up to float rounding of
  W_host + (W_specialist - W_host).
- `save(dir)` writes `meta.json`, one safetensors file per expert, and
  `router.safetensors`. `SkillDeltaNetwork.load(dir, host=...)` restores
  it, with the host passed in or loaded from the saved id.
- `count_params()` returns stored params and params per token. The input
  embedding table counts as a lookup, not as run per token, unless the head
  ties to it (the `wsp.skills_eval.model_params` convention).

## The router

A ridge classifier on the host's own mean-pooled hidden state after layer 4,
over the first 256 tokens of the prompt, always computed with the base
expert active. `fit_router(texts_by_expert)` needs a list of texts for
"base" and for each expert. Features are standardized, targets are one-hot,
and the solve is closed form.

`route(prompt)` returns one expert name. A non-base winner must lead the
runner-up by `base_margin`, else the prompt stays on the base.

Generation routes once from the prompt, then runs with that delta for the
whole sequence. The choice is fixed per sequence, so the KV cache stays
valid. A batch can mix experts. The network groups rows by expert, runs each
group, and puts the rows back in order. Outputs that end early are padded
on the right.

## What round 16d measures

Pair A is Qwen2.5-1.5B-Instruct with the Math and Coder 1.5B instruct
models. Pair B is Qwen2.5-7B-Instruct with Math-7B, Coder-7B, and
DeepSeek-R1-Distill-Qwen-7B. The Hugging Face metadata of R1-Distill
matches the host in layer count, hidden size, heads, MLP width, and vocab
size. Its BPE vocabulary and merges are the same as the host's. Seven
special token names differ. Its rope_theta is 10000.

Cells per pair: the host, each specialist alone (host template, and its own
template as a reference), the routed network at ranks full, 256, 64, 16,
forced-expert cells at ranks 64 and full for each skill, and a random-delta
control at rank 64 with forced routing. Scores are GSM8K 300, MATH-500 200,
MBPP 200, HumanEval 164, MMLU 1000, MedMCQA 1000, all through
`wsp.skills_eval`. Nothing trains except the ridge router.

The go rule: at some rank with stored params at most 1.3x the host, the
routed network is within 2 points of each specialist on its own skill and
within 1 point of the host on MMLU, and the random control is at or below
the host. The forced cells split a miss into router error (routed vs
forced) and compression loss (forced at rank r vs forced full vs the
specialist alone).

Rows also hold router choice shares per benchmark, router accuracy on
held-out domain texts, params stored and per token, decode tokens per second
per expert, and the delta rank statistics.

## Limits

- Same-lineage specialists only. The delta is a difference from the host.
  If the specialist started from another base, D is not small or low-rank
  and carries no skill that transfers. The same shape is not enough.
- R1-Distill-Qwen-7B is distilled from the Qwen2.5-Math-7B base according
  to its model card, not from the Instruct host. Its delta against the
  Instruct host is a delta against a relative.
- Stored params grow with the sum of the factors. At rank 256 with
  embeddings, one 1.5B expert adds about 0.22x of the host. The 1.3x budget
  allows rank 64 for two to three experts, and rank 256 only for one or two.
- The router reads only the prompt. A prompt that mixes domains, or a skill
  with no training text, goes to the base or to the wrong expert.
- Tests run on tiny CPU models only. Nothing in them says a real
  specialist is low-rank.

## Unverified

- Whether real fine-tune deltas are low-rank enough at rank 16 to 256 to
  keep the specialist's score. Round 16d measures it.
- Whether the host's layer 4 mean state separates math, code, and general
  prompts well enough on the real benchmark prompts, and what base_margin
  (0.15 in the round) is right. The router is fit on domain texts that may
  differ in format from the scoring prompts.
- Math versus R1 routing. Both are math. The round gives R1 the MATH train
  problems of level 4 and 5 and the Math expert GSM8K and the easier MATH
  levels. That split is a choice, not a measured fact.
- The 512 and 1024 new-token limits of `wsp.skills_eval` may cut off R1's
  long reasoning, which lowers its score under every arm.
- bf16 on-the-fly deltas differ from a merged bf16 model by rounding. The
  "full" cells use bf16 factors on the GPU for the same reason, so they
  match the specialist alone only up to that rounding.
- Speed. The hooks add two small matmuls per adapted layer. The round
  records tokens per second, and no overhead number exists before it runs.
- Embedding deltas are mined and applied in every cell of the round. Whether
  they matter at low rank is not separated from the other parts.
