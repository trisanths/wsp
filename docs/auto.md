# The Sapien front door

Code: `wsp/auto.py`. Runner: `modal_sapien.py`. Requests: `sapien_requests/*.json`. Tests: `tests/test_auto.py`.

One request goes in. The compiler reads how each source relates to the base, picks a compile method, orders the stages, checks the limits, and builds one network. Other agents mine components (`library/`, `atlas/`). This module takes what they produce and chooses the method.

    sapien plan sapien_requests/qwen15_code_seam.json      # no GPU, no weights
    sapien make sapien_requests/lfm2_tools.json -o out     # small models locally, else prints the Modal command
    modal run modal_sapien.py --request sapien_requests/lfm2_tools.json
    modal run modal_sapien.py --request sapien_requests/lfm2_tools.json --plan-only

`plan` reads the Hub by range reads (config, tokenizer files, safetensors headers, a few small tensor slices). `--facts file.json` plans offline from a table (see `StaticLineage`).

## Request format

    {
      "name": "qwen15_code_seam",
      "base": "Qwen/Qwen2.5-1.5B-Instruct",
      "add": [{"skill": "code", "sources": ["Qwen/Qwen2.5-Coder-7B-Instruct"], "components": null}],
      "remove": [],
      "keep": ["knowledge"],
      "limits": {"max_total_params_b": 2.47, "max_active_params_b": 2.4, "max_heal_minutes": 60},
      "reported": ["google-research-datasets/mbpp", "openai/openai_humaneval", "cais/mmlu"],
      "seed": 0
    }

- `add[].sources` is a list of model ids or `"auto"`. `components` is null or a list of library or atlas ids (catalog rows, used by synth). An optional `method` forces one method; it is refused when the relation does not allow it.
- `remove[].skill` names a skill. `"*"` means every skill except the ones in `keep` (then `keep` must not be empty).
- `limits` are in billions of parameters and minutes. Any of them may be null. `max_active_params_b` counts parameters run per token (the shared rule of `wsp.skills_eval.model_params`).
- `reported` lists the benchmark dataset ids the result will be scored on. The default is `wsp.healdata.BANNED`. No text that trains, fits, selects, or guards comes from a reported benchmark, in any split.
- Validation errors name the field. Unknown fields are errors.

With `"sources": "auto"` the compiler takes models from `atlas/entries`, `atlas/pool_entries` (when present), and `library/models`. It ranks them by atlas skill score, probes up to eight in order, and takes the first one whose relation has a method that fits the limits. When there is no mined data it says which data is missing. Atlas scores and layer maps are measured on benchmarks (`ATLAS_SKILL_DATA` in `wsp/auto.py` lists them). When the benchmark of a skill is in `reported`, its atlas scores do not rank sources (the plan refuses and asks for sources by hand), its layer maps do not pick donor blocks (the depth rule applies), and `best_existing` is not built.

## How a method is chosen

`wsp.lineage.relation(base, source)` classifies the pair from config, tokenizer, safetensors shapes, and cosine on sampled tensors. `wsp.lineage.methods` lists the valid methods for that relation. The plan takes the first one that fits the limits. The fine-tune gate uses the mean and the minimum cosine over layer matrices only (mean above 0.9, minimum at least 0.8), so a frozen embedding cannot lift it. A quantized source (a non-float dtype, or `qweight`, `scales`, or `weight_scale_inv` tensors) is never classed as `same` or `fine_tune` and gets only `whole_model`. `same` means equal weights: relative L2 under 1e-6 on every sampled tensor, never a cosine threshold. A tiny nonzero difference is `fine_tune` with a warning that the difference is small. "Tokenizer same" means the same token-to-id mapping for ordinary tokens (the vocabulary and the merges of `tokenizer.json`, or the vocab files), whatever the chat template, the pad token, or the file formatting (merges as strings or as pairs). A separate flag, `tokenizer_strict_same`, also needs the same template, special tokens, and added tokens. Added tokens that reuse reserved ids set `reserved_ids_reused` and warn. When a card names a gated ancestor (meta-llama), the ancestor lookup tries an ungated mirror (unsloth, then NousResearch); when none can be read the relation stays `fine_tune` with a warning. `feedback` needs the host's tokenizer. `teacher_usable_for_logit_kd` also needs equal added tokens. A method that runs a second model at inference (route to the whole source) is dropped: Sapien builds one network.

| Relation | Methods in order | Why |
|---|---|---|
| same | none | The weights are equal to the base. Nothing to add. |
| fine_tune | weave, delta | Same shapes, same positional base, cosine near 1. Weave mixes per layer at exactly the base size. |
| fine_tune_of_ancestor | weave, delta | Close, but the true parent is another model. The weave reference is that ancestor. |
| diverged_same_shape | synth, feedback | Same shapes, far weights (Qwen2.5 Coder or Math against Instruct). Mixing below the whole model gives chance, so components need translation layers. |
| same_family_other_size | synth, feedback | Another width or depth. A seam with translation layers. |
| moe_same_width | transplant, synth | A MoE of the same family and width as a dense base: routed experts next to each MLP. |
| foreign | pgraft (knowledge, medicine, law skills), synth, feedback | Another family or tokenizer. For a four-option knowledge skill and a dense donor, pgraft comes first: a frozen donor slice runs as a parallel branch of the host, `wsp/pgraft.py`. |
| remove | slim | MoE: keep the top experts per layer for the kept skills. Dense: drop layers, neurons, heads. |

A method is skipped when its size does not fit. Sizes come from configs and parameter counts (`wsp/auto.py` size functions): weave adds nothing, synth and feedback add donor layers plus T_in, T_out, a gate, and the LoRA, transplant adds K experts per layer plus stitches (K and rank shrink until it fits), delta adds a rank-64 difference. When nothing fits, the plan states which limit would allow the smallest option.

### Stage order and chaining

Stages whose output is a plain Hugging Face directory run first: weave, then slim. Each takes the previous output as its base. Weave goes before slim because its members have the full shape. At most one seam stage (pgraft, synth, feedback, transplant, delta) runs last. Two seam kinds cannot be stacked (each replaces the other's branches), so the plan refuses them with that message. Weave also cannot run on a model that already carries seams, because its tensor names would match the branch parts.

## The plan

`plan(request, lineage=...)` returns a dict and never touches a GPU: per source the relation, the methods with fit and reason, the chosen method; the stages with total and per-token parameters after each stage, settings, and heal steps from the minutes limit (a step-time estimate, a guess); the heal text pools with the usable and excluded sources; probe texts (rows 2000 and later of each pool, split a, b, c; heal reads from row 0); practice sets where `healdata.practice` has one; controls per method; comparison arms (base, base + LoRA, base + distillation when a source shares the tokenizer, best existing model when the size limit is far above the base); warnings; refusals.

A skill with no pool in `wsp.healdata` (for example tools) uses the general pool, and the plan says so. A skill with no practice set is verified by probe transfer only.

## The report

`make` writes `out_dir/sapien_report.json`:

- `plan`, `request`, `stages`: each stage's own report (gates, search summary, synth report, heal log, saved path).
- `probe.skills`: per added skill, transfer (the share of the gap between the base and the source that the result closes, on text the source and base both read, split c, never used to select), `content_z` against the random controls where several exist (weave: shuffled differences, measured at the weave stage), and the probe verdict word.
- `probe.drift`: KL from the base to the result on general text. `probe.nll`: text loss on every skill's text, base and final. `probe.practice`: practice scores, base and final.
- `sizes`: measured total and per-token parameters, base and final, checked against the limits. `seconds`.
- `verdicts`: one per component, and `verdict` for the request: met, partly met, or not met, each with reasons. A component is met when transfer is at least 0.3, the probe verdict is "contributes", and drift stays under 0.05 nats. A removal is met when kept skills lose under 0.1 nats and removed skills lose more; with no removed-skill text measured it is partly met. A measured size or time over a limit makes the request "not met".

`score_reported` is a separate call, made once, for the result and each comparison arm. It maps the reported ids to the harness skills (gsm8k, math500, mbpp, humaneval, medmcqa, mmlu) and lists the ones the harness cannot run.

## Backends

`make(request, out_dir, backend)` calls a backend for model loading, text, generation, and saving. `LocalBackend` holds tiny models in memory (tests). `HubBackend` loads from the Hub; `modal_sapien.py` uses it on an H200 with the volume `wsp-models`. The runner plans on the caller's machine and sends the plan and the atlas entries with the request, so the container runs the plan that was checked; `atlas/`, `library/`, and `catalog.jsonl` are also mounted for component rows.

## The pgraft method

`pgraft` writes one slice of a donor from another family into a small host. The plan prints the host layer, the donor span, the Enc and Dec sizes, the data pool, the steps, the params per token after the write-in, and the controls. Layers follow the measured recipe: host layer is host layers // 2 - 1, the donor span is donor layers // 7 to 5 * donor layers // 7 (layers 7 and 4..20 for Llama-3.2-1B with Qwen2.5-7B). `make` skips the text probes (they need a shared tokenizer), loads the host and donor, trains with the code of `wsp/xfam.py`, saves the artifact `<out>/pgraft_v1/` (glue, LoRA, sink vector, and ids of the host and donor), reloads it into a fresh host, and scores the reloaded network on MMLU dev 1000, MMLU held-out 2000, ARC-Challenge test, and OpenBookQA test. The compiler also trains the identity and scrambled-donor controls (setting `pgraft_controls`). Matched LoRA, rank-128 LoRA, and full fine-tune are separate drivers (`modal_wi_xfam.py`). One command from a request: `modal run --detach modal_sapien.py --request demo/requests/xfam_knowledge.json`. Details and numbers: `docs/write_in_compiler.md`.

## What is unverified

- Most methods have not passed on real weights. weave, synth, feedback, transplant, delta, and the front door itself have run on CPU with tiny random models only. Slim has one studio run without a trusted quality number. Every plan prints this as a warning.
- The CPU tests exercise weave, synth, the command line, the arms, and the plan on faked lineage. The feedback and delta runners have a tiny end-to-end test. The transplant runner has none.
- The lineage thresholds (`wsp/lineage.py`) have not been run against the Hub.
- Size estimates for synth, feedback, transplant, and delta come from configs and assume SwiGLU MLPs. The measured size after the build is the one checked against the limits.
- Heal minutes map to steps with a guessed step time. The weave search shrinks population and generations with a guessed candidate time. Nothing stops a heal at N minutes; the verdict flags a run that took longer.
- A source with another tokenizer gets no KL transfer number (its verdict is "partly met"). Transfer by donor-generated text is not wired.
- Weave members need a full local download and about (2 x members + 1) copies of the base in CPU RAM.
- Content_z for seam methods is not computed; synth's own control verdicts are in the stage report.
