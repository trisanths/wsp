# Skill tops

Code: `wsp/tops.py`. Round: `modal_round24.py`. Tests: `tests/test_tops.py`.

A skill top lets one network carry a donor's skill without sending the donor's
output back into the base. The base keeps its bottom layers as a shared trunk.
Each skill gets a top that holds its own stitch, its own upper layers, its own
final norm, and its own head. A router picks one top per sequence.

## The mechanism

```
tokens -> base embedding -> base layers 0..cut-1   (shared, frozen, one KV cache)
                                  |
                  router reads the mean trunk state at the smallest cut
                                  |
        +-------------------------+----------------------------+
        | base top                | skill top                  |
        | base layers cut..end    | stitch (+ donor sink)      |
        | base norm, base head    | donor layers cut_donor..end|
        |                         |   own rotary base          |
        |                         |   private KV cache         |
        |                         | donor final norm           |
        |                         | donor head                 |
        +-------------------------+----------------------------+
                                  |
                            logits, base width
```

Only the chosen path runs for a sequence, so compute per token stays close to
the base. Stored params grow by the size of every top.

A top is added with `SkillTops.add_top(name, donor, cut_host, cut_donor=None,
stitch_kind="mlp", own_head=True, control=None, control_seed=0)`. The donor is
a repo id or a local directory. Its layers, final norm, and head come by byte
range (`wsp/fetch.py`), so the donor's embedding table and lower layers never
download unless the donor is tied (then the head is the donor's embedding
matrix, and it is fetched as the head). The donor layers are the donor's own
decoder classes, run with the donor's own rotary base: Qwen2.5-Math has
`rope_theta` 10000 and the others 1000000, and each top keeps its own.

## Why this follows from the earlier runs

Three results point the same way.

- The stitch study (results/README.md, "Stitch study") ran the base's bottom,
  a trained stitch, and the donor's layers to the end. With the attention sink
  fix and an MLP stitch trained on the donor's own LM loss, the donor loss
  rose by only 0.03 to 0.08 nats at the early cut. A donor can read the base's
  hidden states if it keeps its own upper layers and head.
- Runs 14, 14b, 20, and 22 sent a donor block's output back into the base's
  upper layers, with a LoRA on the base, with a distillation heal, and with a
  cross-tokenizer stitch. None gave a gain a random block or a LoRA of the same
  size did not also give. The base's upper layers cannot read donor features.
- Run 16c swapped a specialist's layers into the base with no stitch. The
  result was chance. Weights from a model pretrained further from the base do
  not drop in.

So the skill keeps its readout. The write-back that failed is not built:
`own_head=False` raises.

## The stitch fit

`fit_stitch(top, texts, steps=800, ...)` runs in this order.

1. The donor's mean state at the first real token is measured on the fit texts
   and stored in the top as the sink. Qwen adds no BOS, so the first token is
   the attention sink and its state does not depend on the input. At the first
   real token of every row (left padding included) the stitch output is
   replaced by this vector, on a prefill only. A decode step never replaces
   anything, because the first token is already in the cache.
2. A ridge fit on paired states starts the stitch (`wsp.stitch.ridge_init`). A
   linear or MLP stitch starts from it. The MLP stitch has a zero output
   layer, so it starts equal to the ridge map.
3. The stitch (and an optional top LoRA) trains on the donor's own next-token
   loss through the top. The trunk and the donor layers are frozen. The
   learning rate is 3e-4 for the MLP and 1e-3 for the linear stitch (the study
   values). A loss that doubles after the first 25 steps stops the fit, restores
   the start, and flags the top.
4. The penalty is the donor's loss through the top minus the donor's native
   loss on the same held-out texts, token weighted, one text per batch. The
   prediction made from the first real token is left out of both when the top
   has a sink, because the sink is a constant and that one token cannot be
   predicted from it. `penalty_all` counts every token, as the study did, and the
   fit report holds both numbers.
5. A penalty above `max_penalty` (default 0.25 nats) flags the top. A flagged
   top is refused: `forward` raises, and the router leaves it out. A research
   run sets `allow_flagged` to score a flagged top anyway, and the row says it
   was flagged.

The optional LoRA (`lora_rank > 0`) goes on the top's donor layers (q, k, v, o,
gate, up, down) and trains together with the stitch. Its params count as stored
params.

## The router

`fit_router(texts_by_class)` regresses one-hot class labels on the trunk's mean
state at the smallest cut among the tops. The state is normed with the next
base layer's input norm (the router in `skill_moe.py` does the same, because
Qwen states have outlier dimensions). The mean runs over the prompt tokens and
leaves out the first real token (the sink). The ridge strength is relative to
the mean covariance diagonal. The logits are 8 times the regression output,
plus a base margin on the base logit. General text maps to class `base`.
`tune_base_margin` picks the smallest margin at which at least 98 percent of
general probe text stays on the base and every top keeps 90 percent of its own
probe text.

Tops may have different cuts. The router reads the smallest cut. The chosen top
then runs the trunk on to its own cut. The network recomputes the first layers
once (the routing pass and the chosen path both run them on the prompt). A
fused version would reuse the routing pass. The cost is a prefill of the
smallest cut's layers, once per call, and nothing per decoded token.

## Forward and generation

`forward(input_ids, attention_mask, top=None)` runs a forced top or the routed
choice. A batch may mix tops: rows are grouped by choice, and each group runs
its own path. `generate(...)` routes once from the prompt, then decodes each
group with a trunk KV cache (base layers up to the chosen cut) and the top's
private cache. Batches may be left padded, calls may repeat. The base group
uses the base model's own `generate`. `generate` is greedy and takes the
arguments `wsp.skills_eval` passes, plus stopping criteria.

Logits always have the base's width. A donor head with more rows loses its
tail (rows no base id reaches). A narrower head is padded with masked rows.
A base token the donor tokenizer lacks has its column masked to the dtype
minimum. Donor-only ids and ids in neither tokenizer pass through as the donor
head gives them. `check_vocab` compares the token to id maps and raises when a
shared token has two ids, when an id names two tokens, or when the head has no
row for a shared id.

## Sizes

`count_params()` returns stored params (the base counted once with a tied head
once, plus every top's stitch, donor layers, norm, head, LoRA, sink, plus the
router) and the params run per token on each path (trunk layers, stitch, top
layers, norm, head). A donor with a tied head stores a copy of its embedding
matrix as the head, and the count says so: the network holds no other use for
it. `sizes_from_configs` computes the same numbers from configs alone, so the
plan needs no weights. Numbers for the round, MLP stitch, two tops for T1 and
T3 and one for T2, router and sinks included:

| Pair | Base | Cut 4 | Cut 8 | Cut 14 | Cut 20 | Cut 24 |
|---|---|---|---|---|---|---|
| T1 stored over base | 1.544B | 2.773 | 2.530 | 2.166 | 1.803 | 1.560 |
| T2 stored over base | 1.544B | 5.003 | 4.400 | 3.494 | 2.588 | 1.984 |
| T3 stored over base | 7.616B | 2.629 | 2.384 | 2.017 | 1.650 | 1.405 |

Per-path params run per token (billions, same at every cut for equal widths):
T1 base 1.544, math and code paths 1.556 (the trunk and the donor layers add
up to one 1.5B model, plus a 12M stitch). T3 base path 7.071, top paths 7.135.
T2 base path 1.544 and the code path falls from 6.37 at cut 4 to 2.64 at cut
24, against a 7.07 donor. (Run per token
counts the head and not the input embedding, as `wsp.skills_eval.model_params`
does; a tied head counts in full.)

Reading the table. At the default cut 1/2 a 1.5B top costs about 0.9B params
(14 donor layers, a 233M head, a 12M stitch), so two tops take the T1 network
to 2.17 times the base. The 1.6 limit of the go rule is met only at cut 24 for
T1 (1.56) and T3 (1.41, and 1.65 at cut 20 misses by 0.05). T2 puts a 7B head
(0.55B) and 7B layers on a 1.5B base, so it cannot go under 1.98 at any cut, and
its limit is 2.0. The cut sweep answers how small a top can stay within 3 points
of its donor, and the stored limit decides which of those cuts the go rule can
use. The routed cell at the default cut cannot meet the stored limit, so the
round adds one routed cell per pair at cut 24, one seed.

## What the round measures

Round 24 (`modal_round24.py`) scores, per pair and cut:

- forced-top cells per skill: stitch penalty, the skill's benchmarks, MMLU;
- at the default cut: the routed network over every benchmark with router
  choice shares per benchmark, the scrambled-top control, the forced top with a
  rank 16 top LoRA trained 800 steps with the stitch, and (T1) the forced math
  top with the donor's own chat template;
- reference cells: the base, each donor, and a LoRA on the base distilled from
  the donor on the same texts and steps, with the rank set so the LoRA has as
  many params as the stitch.

The go rule is in the script's docstring. The round also reports the deepest cut
at which the forced top stays within 3 points of its donor.

Two seeds go to the default cut and the controls. The other four cuts have one
seed. A seed changes the batch order, the LoRA init, and the scramble. The fit
texts and the ridge init do not change, and decoding is greedy.

## Limits

- A top with its own head needs the base tokenizer's ids. Cross-tokenizer
  donors are out. The token to id check is the gate.
- One path per sequence. A sequence that mixes skills gets one top. There is no
  per-token routing, so a prompt that needs math in one place and code in
  another is served by one of them.
- Stored params grow by every top, and a top at a shallow cut is most of a
  model. The saving is compute per token, not storage.
- The donor must have a dense decoder stack of one layer class per range. Models
  with two rotary tables (Gemma 3) or with sparse layers are not wired.
- `own_head=False` is not built.
- The generation loop is Python over layers. It is slower than the base model's
  fused path. Tokens per second in the rows are measured, not assumed.

## Not verified

- Nothing in `wsp/tops.py` has run on a real model. The tests use tiny random
  models on CPU. The smoke cell is the first real run.
- That the Math and Coder tokenizers carry the same ids as the Instruct
  tokenizers. `check_vocab` raises at add time if not, and prints what it masks
  when the sets differ.
- That the penalty at each cut matches the study's 0.03 to 0.08. The study used
  the Math-1.5B donor into the 1.5B base at layers 8 and 14 only.
- That the router separates MATH-500, HumanEval, MBPP, and MMLU prompts. It fits
  on healdata prompts, and the benchmark prompts have other formats.
- That the deep cuts keep a skill. At cut 6/7 the top is four donor layers and
  a head, and the stitch has to carry most of the donor's computation.
- The GPU hours and memory in the plan are estimates, not measurements.
