# MLP-cluster grafts: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Graft an MLP neuron cluster from a donor of any family into a host, heal the glue, and measure it against a random-cluster control.

**Architecture:** Fetch only the donor tensors a part needs. Build the part from the donor's own Transformers MLP class. Wrap it in a graft (W_in, part, zero-init W_out) on a forward hook at a host layer. Initialize W_in by ridge regression on offset-matched hidden states. Train only the glue.

**Tech Stack:** Python 3.12, PyTorch, Transformers 4.57, safetensors, huggingface_hub, Modal.

**Spec:** `docs/superpowers/specs/2026-11-14-mixed-family-compiler-design.md`

## Global constraints

- Plain, readable Python: simple loops and ifs; no dense comprehensions, `setdefault`, or heavy type annotations.
- Local runs: CPU tests on tiny random models only. Real models run on Modal profile `trisanth`.
- A freshly compiled graft model gives the host's exact logits (W_out starts at zero).
- Only glue parameters train. Host and part stay frozen.
- Every result reports the random-cluster control with identical heal settings.
- Search split: 200 items from GSM8K train. Heal data: the rest of GSM8K train. Held-out: GSM8K test, not used in this plan.

## Review focus

1. Host decoder layers that return a tuple vs a tensor: the hook must handle both.
2. Donors whose MLP norm is `pre_feedforward_layernorm` (Gemma) vs `post_attention_layernorm` (Llama, Qwen).
3. Sharded donors (`model.safetensors.index.json`) vs a single `model.safetensors`.
4. bf16 tensors in a fetched shard (safetensors dtype `BF16`).
5. No matched offsets between tokenizers: the fit must fall back to random W_in and say so.

---

### Task 1: Tensor fetch by byte range

**Files:** Create `wsp/fetch.py`. Test `tests/test_fetch.py`.

**Interfaces:**
- Produces: `read_header(read_range) -> (header_dict, data_start)`, `read_tensors(read_range, names) -> dict[str, Tensor]`, `fetch_tensors(repo_id, names, cache_dir=None) -> dict[str, Tensor]`. `read_range(offset, length) -> bytes`.

- [ ] Test `test_read_tensors_from_local_file`: save a safetensors file with an f32 tensor and a bf16 tensor; read both through a local `read_range`; assert equal to the originals.
- [ ] Test `test_read_only_requested_names`: request one of three names; the result has one key.
- [ ] Implement. `fetch_tensors` reads `model.safetensors.index.json` if present (else `model.safetensors`), groups names by shard, and reads each shard's header and tensors with HTTP `Range` requests on `hf_hub_url`. Send `HF_TOKEN` when set.
- [ ] Run `pytest tests/test_fetch.py -q`. Expect PASS. Commit.

### Task 2: MLP-cluster part, graft, and hooks

**Files:** Create `wsp/parts.py`, `wsp/graft.py`. Test `tests/test_graft.py`.

**Interfaces:**
- Consumes: `fetch_tensors` (Task 1), or any `get_tensors(names) -> dict`.
- Produces:
  - `mlp_tensor_names(config, layer) -> list[str]` and `build_mlp_cluster(config, layer, indices, tensors) -> nn.Module` in `wsp/parts.py`. The module maps a donor-size hidden state to the cluster's donor-size MLP output, including the donor's own pre-MLP norm (and Gemma's post-feedforward norm).
  - `Graft(d_host, d_donor, part)` in `wsp/graft.py`, with `w_in`, `part`, `w_out` (zeros) and `forward(h) -> h + w_out(part(w_in(h)))`.
  - `attach_grafts(host_model, placements) -> handles`, where `placements` is a list of `(host_layer, Graft)`. Stores grafts in `host_model.wsp_grafts` (an `nn.ModuleList`).
  - `glue_parameters(host_model) -> list[Parameter]`.

- [ ] Test `test_cluster_equals_partial_mlp`: tiny random Qwen2 donor; with all neurons selected, the part's output equals the donor layer's MLP applied to its normed input.
- [ ] Test `test_fresh_graft_keeps_host_logits`: tiny random Llama host (d=64) plus a Qwen2-donor cluster (d=48); logits equal the host's exactly.
- [ ] Test `test_gradients_reach_glue_only`: one backward pass; `w_out` has a nonzero grad, host and part params have none.
- [ ] Implement. Build the donor model on the `meta` device from its config to find the MLP class, its module path, and its norms; instantiate the MLP with `intermediate_size = len(indices)` and load sliced tensors (slice the dimension equal to the original intermediate size).
- [ ] Run `pytest tests/test_graft.py -q`. Expect PASS. Commit.

### Task 3: Alignment fit, heal, program, compile, and load

**Files:** Create `wsp/align.py`, `wsp/heal.py`, `wsp/extract.py`, `wsp/load.py`. Modify `wsp/program.py`, `wsp/compiler.py`, `wsp/__init__.py`. Test `tests/test_align.py`, `tests/test_mixed_compile.py`.

**Interfaces:**
- Produces:
  - `match_offsets(host_ends, donor_ends) -> list[(i, j)]`; `ridge_fit(x, y, lam) -> (w, r2)` with `w` shaped `(d_donor, d_host)` and r2 measured on a 10% holdout.
  - `heal(model, tokenizer, texts, steps, lr, batch_size, seq_len, seed) -> list[float]` (loss per step).
  - `top_activation_neurons(model, tokenizer, texts, layer, n) -> LongTensor` in `wsp/extract.py` (the stub extractor).
  - `Program(host=..., base=alias)`, `.graft(component_id, at=None, control=None)`, `.heal(steps, data, lr=3e-4)`; old `program.json` files still load.
  - Catalog type `mlp_cluster` with fields `model`, `layer`, `units`.
  - `wsp.load(path) -> (model, tokenizer)`.

- [ ] Test `test_match_offsets`: host ends `[3, 7, 12]`, donor ends `[3, 5, 7, 12]` give `[(0, 0), (1, 2), (2, 3)]`.
- [ ] Test `test_ridge_recovers_linear_map`: y = x @ A.T exactly; fit gives r2 > 0.99.
- [ ] Test `test_program_round_trip_with_grafts`: a program with masks, a graft with `at=3`, a control graft, and heal round-trips through JSON; an old `{"base": ...}` dict loads.
- [ ] Test `test_compile_and_load_tiny`: compile a tiny host plus a tiny donor cluster (local directories, no network) with 2 heal steps; `wsp.load` returns a model whose logits equal the compiled model's.
- [ ] Implement. `control="random"` replaces the cluster's indices with random indices of the same count from the same donor layer (seeded). Compile saves `graft_weights.safetensors`, `grafts.json` (donor id, layer, indices, host layer, d sizes), `program.json`, `compile_report.json` (r2 per graft, glue and part param counts, heal losses), and host weights only when masks or task vectors changed them.
- [ ] Run `pytest -q`. Expect all PASS. Commit.

### Task 4: Modal run on real models

**Files:** Modify `modal_app.py`. Output `outputs/graft_run_1.json`.

- [ ] Add `graft_experiment` on one H100: host `unsloth/Llama-3.2-1B-Instruct`, donor `Qwen/Qwen2.5-Math-1.5B`, donor layer 18, host layer `round(18 / 28 * 16)`, 512 top-activation neurons from GSM8K train text.
- [ ] Compile three models: host with no graft, random-cluster control plus heal, real cluster plus heal. Heal 300 steps on GSM8K train minus the search split.
- [ ] Score each on the 200-item search split with greedy decoding and final-number match. Label it an interim search-split check; Person B's eval replaces it.
- [ ] Record accuracy, r2, glue params, part params, heal loss, and wall-clock. Commit and post the results.
