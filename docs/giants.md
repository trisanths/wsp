# Giant open MoE models as component sources

Date of checks: 2026-11-14 (saliency review fixes: 2026-11-15). Code: `wsp/giant.py`, `modal_giant.py`, `tests/test_giant.py`, `scripts/giant_saliency_tf5.py`. Zoo rows: `atlas/zoo.json` (field `stream`).

No machine we can rent holds a giant. Sapien needs two things from one: which experts matter for a skill (the atlas expert map and the saliency map), and the weights of the chosen experts. `wsp/giant.py` gets both by reading the repo with HTTP range requests. It never downloads a whole model.

## Repos that exist

All checked against the Hugging Face API (model info, `config.json`, `model.safetensors.index.json`, shard headers). Sizes are the sum of the `.safetensors` files.

| Repo | Layers | Hidden | Routed experts | Top-k | Expert size | Router | Stored as | Files | Repo size | License |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V4-Flash | 43 | 4096 | 256 + 1 shared | 6 | 2048 | sqrtsoftplus, bias, first 3 layers hash by token id | FP8 dense, FP4 experts | 46 | 159.6 GB | MIT |
| deepseek-ai/DeepSeek-V4-Pro | 61 | 7168 | 384 + 1 shared | 6 | 3072 | same | same | 64 | 864.7 GB | MIT |
| zai-org/GLM-5.3-Flash | 45 (3 dense) | 4096 | 288 + 1 shared | 8 | 2048 | sigmoid, bias | FP8 mlp, BF16 attention | 62 | 328.3 GB | MIT |
| moonshotai/Kimi-K3 | 93 (1 dense) | 7168 | 896 + 2 shared | 16 | 3072 at width 3584 | sigmoid, bias, latent MoE | mxfp4 experts, BF16 rest | 96 | 1560.9 GB | kimi-k3 (other) |
| moonshotai/Kimi-K3.1-Pro | 93 | 7168 | 896 + 2 shared | 16 | same | same | same | 96 | 1561.3 GB | kimi-k3.1-pro (other) |
| Qwen/Qwen3.8-2.4T-A95B | 92 | 8192 | 512 + 1 shared | 10 | 2048 | learned top-k gate | BF16, experts fused per layer | 213 | 4892.4 GB | qwen3.8-max (other) |
| Qwen/Qwen3.5-397B-A17B | 60 | 4096 | 512 + 1 shared | 10 | 1024 | learned top-k gate | BF16, experts fused per layer | 94 | 806.8 GB | Apache-2.0 |

Two names from the request:

- "Kimi K3.1 pro" exists: `moonshotai/Kimi-K3.1-Pro`, created 2026-11-02. It has the shapes of K3 plus a sparse-attention indexer (`self_attn.indexer.*`) in the 24 full-attention layers. The repo ships configuration code (`configuration_kimi_k3.py`, `auto_map` has only `AutoConfig`) and no modeling file, and transformers 5.23.0 has no `kimi_k31` class. A layer cannot be built from it today. The zoo row has `stream.supported = false`.
- "Qwen Max around 278B" does not exist as open weights. The only open Max is `Qwen/Qwen3.8-2.4T-A95B` (license name `qwen3.8-max`; its README calls Qwen3.8-Max the hosted version of it): 2.45T parameters, 95B active. The other large Qwen repos are Qwen3.5-397B-A17B (403B), Qwen3-Coder-480B-A35B and Qwen3-235B-A22B. Nothing is near 278B. Qwen3.8-Flash-Next is 180B.

### Tensor names and storage, one layer

- DeepSeek-V4 (checkpoint uses the native names, not the transformers names): `layers.L.ffn.experts.E.{w1,w2,w3}.weight` is `I8` with shape (I, H/2) for w1 and w3 and (H, I/2) for w2, two FP4 e2m1 values per byte, low nibble first. `.scale` is `F8_E8M0` with shape (rows, cols*2/32). Router: `layers.L.ffn.gate.weight` (E, H) and `.bias`; layers 0 to 2 have `ffn.gate.tid2eid` (vocab, 6, `I64`) instead of a bias. Dense weights (`attn.wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`, shared experts) are `F8_E4M3` with a `.scale` of 128x128 blocks. Read from real shard headers: V4-Flash expert w1 is `I8` [2048, 2048], scale `F8_E8M0` [2048, 128].
- GLM-5.3: `model.language_model.layers.L.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight` is `F8_E4M3`, with `.weight_scale_inv` `F32` of 128x128 blocks. Read from a real header: gate_proj [2048, 4096] with scale [16, 32]. Router `mlp.gate.weight` and `mlp.gate.e_score_correction_bias`. Attention is BF16 (6.9B BF16 parameters in total).
- Kimi K3: `language_model.model.layers.L.block_sparse_moe.experts.E.{w1,w2,w3}.{weight_packed,weight_scale}`, both `U8`. Read from a real header: w1 packed [3072, 1792], scale [3072, 112] (one scale per 32 inputs of width 3584). `routed_expert_down_proj` and `routed_expert_up_proj` map 7168 to 3584 and back, so experts run in the latent width. The router reads the full 7168.
- Qwen3.8 and Qwen3.5: `model.layers.L.mlp.experts.gate_up_proj` (E, 2I, H) and `down_proj` (E, H, I), one tensor each, BF16. Expert rows are contiguous, so one expert is one byte range.

### Which transformers loads them

| Repo | Class | Version |
|---|---|---|
| DeepSeek-V4-Flash, Pro | `DeepseekV4ForCausalLM` | native in 5.23.0 (not in 4.57.1) |
| GLM-5.3-Flash | `Glm5NextForConditionalGeneration` (text: `Glm5NextTextModel`) | native in 5.23.0 |
| Qwen3.8-2.4T-A95B | `Qwen3_5MoeForCausalLM` | native in 5.23.0 |
| Qwen3.5-397B-A17B | `Qwen3_5MoeForConditionalGeneration` | native in 5.23.0 |
| Kimi-K3 | remote code `KimiK3ForConditionalGeneration` | needs 4.57.x. The code imports `OutputRecorder` and `check_model_inputs`, which 5.23.0 does not have. It also needs `einops` and `fla-core`, and forces `flash_attention_2`. |
| Kimi-K3.1-Pro | none | the repo has `configuration_kimi_k3.py` but no modeling file |

The zoo row for Kimi-K3 said `5.x`. It now says `4.57.1`.

Tokenizers: V4 and GLM ship `tokenizer.json`; Kimi ships `tiktoken.model` with `tokenization_kimi.py` (remote code); Qwen ships `tokenizer.json`.

## What the code does

### Streaming profile

`giant.profile_streaming(source, config, tokenizer, skill_texts, general_texts, ...)` returns `{"maps": {skill: map}, "info": {...}}`. Each map has the format of `wsp.atlas.expert_skill_map`: `layers[layer]` with `skill_freq`, `general_freq`, `log_ratio`, `top`, plus `top_k` and `num_experts`. `atlas.make_entry` and `atlas.catalog_rows` take it unchanged. With saliency on (the default) each layer also has `skill_saliency`, `general_saliency` and `top_saliency`; see the Saliency section.

Per layer: read the layer's tensors by byte range (merged into few requests), build the layer with the transformers class on the device, decode FP8, FP4 and mxfp4 on the device, write the experts into the layer, run every cached hidden state through it, hook the router, keep the output states, free the layer. The next layer's tensors are fetched in a background thread while the current one runs.

How the layer gets the same inputs as in the full model: the model is built on the meta device with each decoder layer replaced by a stub. The model's own `forward` runs per text and builds the masks, position embeddings and hash-routing inputs. Stubs before the active layer pass their input through. The active stub injects the cached state (for Kimi, the hidden state and the block residual), runs the real layer, stores its output and stops the forward. Models whose forward changes the hidden state between layers are not handled.

Resume: `state_dir` saves router counts (and saliency sums) after every layer and hidden states every few layers. A rerun with the same texts continues from the last saved states. A rerun of a finished state runs no layer and returns the whole result, with `top_k`, `num_experts` and the saliency method taken from the saved files. Each file is written under a temporary name and then renamed, so a container that stops during a write leaves the older file. A file that does not load (older code wrote in place) is skipped with a log line, and the run continues from the newest checkpoint whose files all load.

Layer range. `max_layers` counts from layer 0. The skeleton knows from the meta model which layers have a router (`Skeleton.routed`), so a range without any MoE layer stops with a ValueError that names the first MoE layer, before the embedding or a layer is read. Checked on the real configs with transformers 5.23.0: GLM-5.3-Flash has 42 MoE layers, the first is layer 3, and a 2-layer run stops with that message after 0 bytes. DeepSeek-V4-Flash has a router in all 43 layers (layers 0 to 2 are the hash routers).

### Saliency

Definition. For one MoE layer, one expert e and one set of texts: saliency[e] is the sum, over the tokens whose router picks e, of the routing weight the model gives e on that token (after its own top-k, renormalization and scaling factor) times the L2 norm of the output vector of e on that token. It measures how much the expert writes into the residual stream on that text. The pick rate measures only how often the router chooses it. `wsp.slim.collect_moe_stats` is the reference definition, and the streamed numbers equal it.

Why. Round 19 of the compiler kept 32 of 128 experts per layer of Qwen3-30B-A3B with no heal. Chosen by saliency on code text, the pruned model scored HumanEval 57.9 and MBPP 55.3 (full model 91.5 and 84.4). Chosen at random, or by the ratio of skill pick rate to general pick rate, it scored 0.0 on both. Saliency is the ranking that survived, so the giant profiles carry it next to the pick rates.

Fields, per skill map and layer, next to the pick-rate fields (which do not change):

- `skill_saliency` and `general_saliency`: one float per expert, the saliency on the skill texts and on the general texts divided by the number of tokens of that set, so sets of different size compare. Seven significant digits.
- `top_saliency`: up to `top` experts with the highest skill saliency, best first. Experts with saliency 0 (never picked on the skill texts) are left out. Ties go to the expert that is picked more often, then to the lower id, as in `slim.rank_experts`.

`info["saliency_method"]` names the method that ran, and `info["saliency_norm"]` the norm whose RMS divides the gains (`routed_expert_norm` for Kimi-K3, None for the others). Helpers: `giant.compact_maps(result)` returns the two compact maps for an atlas entry, `{"expert_maps": {skill: {layer: [ids]}}, "expert_saliency_maps": {skill: {layer: [ids]}}}` (the first from `top`, the second from `top_saliency`). `giant.saliency_table(result)` returns the saliency of every expert, and `giant.top_by_saliency(values, top=n)` ranks a layer of it for any n.

How the output of one expert is obtained, per family. The forward of a model never shows one expert's output next to its routing weight, so the saliency pass computes the outputs once more.

| Family | Experts in the layer | Method |
|---|---|---|
| DeepSeek-V4, GLM-5.3, Qwen3.5 and Qwen3.8 (transformers 5.x) | one fused module, called as `experts(hidden, top_k_index, top_k_weights)`, returns only the weighted sum | `fused`: a hook on the module reads the indices and weights. One more call of `module.forward` on tokens x top_k rows: row j holds the hidden state of token j // top_k, the single index of slot j % top_k, and weight 1, so output row j is the output of that expert on that token. |
| Kimi-K3 (remote code) and the transformers 4.x families (qwen3_moe, olmoe, mixtral, deepseek_v2, deepseek_v3) | a ModuleList | `list`: the pass calls each picked expert on the rows of its tokens, as `slim` does. The weights come from the router output: taken as they are when the router returns them, or softmax, top-k and renormalization as in `slim.route` when it returns logits. For Kimi-K3 each gain is then divided by the RMS of the token's routed sum (next paragraph). |

A hook on each expert of a ModuleList would not do: the block hands an expert its rows in its own order (by top-k slot in Qwen3-MoE and Mixtral, by an unstable argsort in DeepSeek-V2 and Kimi), so a hooked output cannot be paired with the routing weight of its token. On the first text of every layer the list method compares its expert input with the rows one expert really received, and stops if they differ. Both methods stop if the saliency pass saw fewer tokens than the router.

Kimi-K3: latent width and a norm after the experts. `config.json` sets `routed_expert_hidden_size` 3584 and `latent_moe_use_norm` true, and the index has `routed_expert_down_proj.weight`, `routed_expert_norm.weight` and `routed_expert_up_proj.weight` in all 92 MoE layers. `KimiSparseMoeBlock.forward` (read on the Hub, not run) does this per token: the experts read `routed_expert_down_proj(hidden)`, their outputs are summed with the routing weights into y, then `y = routed_expert_norm(y)`, a `KimiRMSNorm` (`weight * y / sqrt(mean(y^2) + eps)`), then `routed_expert_up_proj(y)`. So what the routed experts of a token write has a fixed size, however large their outputs are, and weight times output norm alone would rank experts by a size the model removes. The saliency pass therefore reads the input of `routed_expert_norm` (named in `KimiLayout.expert_norm`) and divides the gain of every (token, expert) pair by the RMS of that token's y, the number the norm divides by. Kimi saliency is therefore the L2 norm of the expert's share of the normalized sum, in the latent width of 3584, before the learned weight of the norm and before the up-projection. Those two maps are the same for every expert, but they are not part of the number. On the first text of every layer the pass checks that the output of the norm is `weight * y / rms(y)`, and stops if it is not. The transformers 5.23.0 blocks of DeepSeek-V4, GLM-5.3 and Qwen3.5-MoE add the output of the experts module to the shared-expert output as it is (read in the source), so their numbers do not change.

Self-checks, all on the first text of every MoE layer, all with the advice to run with `--no-saliency` when they fail. The fused method adds its recomputed rows up again, weight times output over the slots of each token, and stops unless the sum is the tensor the experts module returned (`check_fused`). The list method checks the expert input and, for Kimi, the norm. The limit is a difference of 1e-4 of the largest value in float32 and 5e-2 in the 16-bit float types (`giant.check_tolerance`); a wrong pairing of tokens and experts gives a difference of the size of the values. Measured on CPU with tiny random weights: on the real transformers 5.23.0 blocks (DeepSeek-V4, GLM-5.3, Qwen3.5-MoE; eager, grouped_mm, batched_mm) the fused check is at 0 to 8.6e-8 in float32 and at 2.2e-3 to 5.6e-3 in bfloat16; on the toy giants in bfloat16 it is under 1e-2. A saliency value that is not finite stops the run with the layer and the text set in the message, because one inf or nan would spoil the ranking of the whole layer and would go into the JSON files.

Log. With saliency on, every MoE layer logs its method, the self-checks that passed with their measured difference, and one line per text set from `giant.saliency_report`: the tokens, the saliency sum and the sum per token, the top 3 experts by saliency next to the top 3 by pick count, how many of the top `top` by saliency are also in the top `top` by picks, and how many picked experts have saliency 0. A working pass gives 0 for the last number, since a picked expert has a positive weight and an output.

Cost, from the code. Per text and MoE layer the layer's own forward evaluates tokens x top_k (token, expert) pairs. The saliency pass evaluates the same pairs once more, grouped by expert in the same way: one extra call of the fused module, or one call per picked expert. So the routed experts run twice. Attention, router, shared experts and norms run once, and nothing is downloaded or decoded again. The forward work of a layer is multiplied by less than 2. `giant.SALIENCY_COMPUTE_FACTOR` is 2.0, and `modal_giant.estimate` multiplies the compute hours by it, as an upper bound, since the assumed seconds per layer also cover the decode. No seconds have been measured.

Extra memory, by arithmetic and not measured. The fused method holds the repeated hidden states and the recomputed outputs, two tensors of tokens x top_k x hidden values for one text (42 MB each in bf16 for 256 tokens at the Qwen3.8 shape of 10 slots and width 8192). During the extra call the experts module allocates its own tensors on those rows, as it does in the layer's own call. In `grouped_mm_experts_forward` of transformers 5.23.0 these are about five more of that size (the sorted hidden states, the projection output, the weighted output, its reordered copy and the result; read from `transformers/integrations/moe.py`). The norms are taken in float32, 256 rows at a time (`giant.row_norms`), so the outputs get no float32 copy: on CPU the torch profiler shows one block as the largest allocation. `torch.linalg.vector_norm(..., dtype=torch.float32)` is not used, because on CPU it allocates the whole float32 copy (same profiler). The list method holds the output of one expert and, for Kimi, one float32 copy of the routed sum (tokens x 3584). Against a decoded layer of 12.9 to 59.2 GB this is small.

Off switch. `profile_streaming(..., saliency=False)` and `modal_giant.py --no-saliency` run the pick counts alone. The result and the state files are then the same as before saliency existed. Use it when the saliency pass fails on one family.

Resume and old state. A run with saliency writes into the per-layer file of the counts: the saliency sums, `giant.SALIENCY_STATE_VERSION` (now 2), and the method and norm of the pass. A rerun with saliency stops with a ValueError when the saved layers have no saliency (older code or a run with saliency off wrote them) or have saliency of another version (other saliency code wrote them; the first saliency code wrote no version). It also stops when the saved method differs from the method of the first new MoE layer. The first two checks run before the model skeleton is built, so nothing is downloaded. The message gives the two ways out: run with `--no-saliency` to finish with pick counts alone, or delete the state (`modal_giant.py --fresh`, which calls `giant.clear_state`) and measure saliency in every layer again. A run with saliency off can continue any state. The version must go up whenever the saliency code changes what the sums mean. A change that keeps the version would be continued without a message, so a repeat of a smoke after a code change needs `--fresh`.

### Single experts

`giant.fetch_expert(source, layer, expert_ids, layout=...)` returns `gate`, `up`, `down` (decoded, or stored with their scales when `decode=False`), `router_rows` and `router_bias` for the chosen experts, by byte range.

Router rows of the top saliency experts. The compiler needs the router rows of the experts it transplants. `fetch_expert` returns them for every layout (plain per-expert, plain fused, DeepSeek-V4, GLM-5.3, Kimi):

```python
picks = giant.compact_maps(result)["expert_saliency_maps"][skill][str(layer)]   # or entry["expert_saliency_maps"][skill][str(layer)]
part = giant.fetch_expert(source, layer, picks, layout=layout)
part["router_rows"]   # (len(picks), hidden): row i is the router weight of expert picks[i]
part["router_bias"]   # (len(picks),) float32, or None when the router has no bias
part["gate"], part["up"], part["down"]   # the expert weights, in the same order
```

The router tensors are `ffn.gate.weight` and `ffn.gate.bias` in DeepSeek-V4, `mlp.gate.weight` and `mlp.gate.e_score_correction_bias` in GLM-5.3, `block_sparse_moe.gate.weight` and `.e_score_correction_bias` in Kimi-K3, and `mlp.gate.weight` (no bias) in Qwen3.5 and Qwen3.8. In the index files of V4-Flash, GLM-5.3 and Kimi-K3 no scale tensor sits next to them, so they are read as stored. DeepSeek-V4 layers 0 to 2 pick experts with the table `ffn.gate.tid2eid`, which `fetch_expert` does not return; their router rows only weight the picked experts. For Kimi the router row has the full width of 7168 while the experts have the latent width of 3584. `giant.expert_download_bytes(source, layout, {layer: [ids]})` reports the bytes from the shard headers. Measured on the real repos: one V4-Flash expert is 13.4 MB (fp4 plus scales), one GLM-5.3 expert 25.2 MB, one Kimi-K3 expert 17.6 MB. Eight experts in every MoE layer of V4-Flash are about 4.6 GB, against 155 GB for the model.

## Verification

Exactness (CPU, `tests/test_giant.py`, 109 tests, transformers 4.57.1):

- Streamed expert maps equal `atlas.expert_skill_map` on the fully loaded tiny model, exactly (same floats), for qwen3_moe, olmoe, mixtral, deepseek_v2 and deepseek_v3, saved as safetensors. A dense layer (qwen3_moe with `mlp_only_layers`, deepseek first-k-dense) gives no entry, as in atlas.
- Streamed hidden states equal `output_hidden_states` of the full forward after 2 layers (`torch.equal`).
- `fetch_expert` equals direct state-dict reads for per-expert and fused checkpoints, with router rows and bias. `expert_download_bytes` equals the bytes read.
- Toy giants written in the V4, GLM and Kimi checkpoint formats (FP8 with block scales, FP4 with e8m0, mxfp4 as `weight_packed`, hash routing, q/k/v conv concat, a two-state layer with a block residual, extra attributes on layers) match the full model. These toys run in transformers 4.57.1 with custom classes. The Kimi toy has the block of the real model: hidden width 48, experts at the latent width 32 behind `routed_expert_down_proj`, then `routed_expert_norm` (an RMS norm with the math of `KimiRMSNorm`) and `routed_expert_up_proj`.
- Resume, prefetch and refusal of dense models.
- Saliency. Streamed `skill_saliency` and `general_saliency` equal `slim.collect_moe_stats` per layer and expert on tiny qwen3_moe, olmoe, mixtral and deepseek_v3, within a relative 2e-6 (the largest difference seen is 4.4e-7, float32 rounding), and `top_saliency` equals the head of `slim.rank_experts("saliency", ...)`. `slim` has no entry for deepseek_v2 or for the toy giants, so those (deepseek_v2, and the V4, GLM and Kimi toys, hash routing included) are compared with a computation written in the test that takes one token and one top-k slot at a time. In every case the pick-rate fields still equal `atlas.expert_skill_map`.
- Kimi norm. The Kimi toy runs through `profile_streaming` with the down-projection and the norm, and its saliency equals the direct computation that divides by `sqrt(mean(y^2) + eps)` of each token's routed sum. In every layer it differs from weight times output norm alone by more than 1 percent for at least one expert, so the test would notice a pass that skips the division. A block whose norm is a LayerNorm is refused.
- Self-checks and bad values. A fused module whose output depends on the batch is refused by `check_fused`. Both self-checks pass in bfloat16 on the toy giants at less than a fifth of the limit. An inf in one expert weight stops the run with the layer and the text set in the message, and the run with saliency off still finishes.
- Log. Every MoE layer and text set has a `saliency_report` line whose sum equals the result, and a run with saliency off logs no saliency line.
- A layer with two experts whose saliency is worked out by hand (0.6 x 10 x sigmoid(1) and 0.4 x 4 x sigmoid(2)) gives those numbers. Both experts have pick rate 1.0 there, so only saliency tells them apart.
- With saliency off the maps equal the atlas maps as whole dicts and the state files have the old two fields. A run stopped after two layers and resumed gives the same floats as a run that never stopped (qwen3_moe, GLM toy, Kimi toy). State without saliency is refused with the message above, and so is a state that has saliency in some layers only. The refusal comes before the skeleton is built and after 0 bytes read.
- State of other saliency code. Sums without a version (what the first saliency code wrote) or with another version are refused, also for a rerun of the same layer range, and sums of another method are refused at the first new MoE layer. After `clear_state` every layer runs again.
- A rerun of a finished state runs no layer and returns the same maps with `top_k` 2, `num_experts` 6, the method and, for the Kimi toy, the norm, with saliency on and off.
- Torn files. With the newest hidden state file cut in half, or a counts file under a good checkpoint cut in half, the rerun logs the file, continues from the checkpoint before it, and gives the same maps. A save that stops halfway leaves the older file.
- A layer range without a MoE layer (deepseek_v3 with `max_layers=1`) stops with the first MoE layer in the message after 0 bytes read.
- `modal_giant.py` without Modal: `run_options` carries every flag; `run_profile` (on a tiny model, with the volume paths in a temporary folder) passes `--top`, `--max-layers`, `--no-saliency` and `--fresh` through, stores and merges the saliency table, keeps the stored table in a run with saliency off, and commits the volume when the profile fails; `save_local` writes the entry, the results file and the printed row.
- The list method reads the projected expert input of a latent MoE block and stops when the layout does not name the projection. The fused method skips the "no expert" index that transformers 5.x uses.
- `fetch_expert` on the `top_saliency` experts returns their router rows and bias for the V4, GLM and Kimi toys.

Checks against the real classes and real key names, run in a scratch environment with transformers 5.23.0 (not part of the default test run):

- Real index key names of V4-Flash (43 layers) and V4-Pro (61), GLM-5.3 (45), Kimi-K3 (93) and Qwen3.8 (92) and Qwen3.5-397B (60), mapped through the layouts, give exactly the state-dict keys of the transformers layer, in every layer. Kimi-K3.1-Pro differs in 24 layers (the indexer keys).
- End to end on tiny random models of the real classes: a V4 model with all 43 layer types (compressor, indexer, hash routing, hyper-connections) written as a native checkpoint with FP8 dense and FP4 experts, a GLM-5.3 model with 45 layers written with FP8, and a Qwen3.8 model with fused experts. In each, the streamed expert map equals `expert_skill_map` of the full model with the decoded weights. These runs were made before saliency existed and have not been repeated with it.
- Saliency on the real MoE blocks: `scripts/giant_saliency_tf5.py` (CPU, tiny random weights) builds `DeepseekV4SparseMoeBlock` (hash router and learned router), `Glm5NextTextMoE` and `Qwen3_5MoeSparseMoeBlock`, and compares `giant.SaliencyMeter` with a pair-by-pair reference from the block's own parameters in float32. The largest relative difference is 1.1e-7 (run of 2026-11-15, transformers 5.23.0, torch 2.7.1), with the eager, grouped_mm and batched_mm experts implementations, and the block output is the same with the meter on. The script also prints the difference that `check_fused` measured, in float32 and with the block in bfloat16 (numbers in the Saliency section). This checks the saliency pass on one block, not a whole streamed model.
- One whole tiny model of a real class with saliency: a 3-layer `Qwen3_5MoeForCausalLM` (grouped_mm experts), saved as safetensors and streamed with a stop after 2 layers and a resume, in a scratch script that is not in the repo (2026-11-15). In float32 the streamed saliency equals a pair-by-pair reference of the full model within a relative 2.7e-7 and the pick-rate fields equal `expert_skill_map`. In bfloat16 the self-check is at 2.2e-3 to 3.2e-3 in the 3 layers, single experts differ from the float32 reference by up to 9 percent, and `top_saliency` (3 experts) is the same as in float32 in all 3 layers. No whole V4 or GLM model has been streamed with saliency.
- The skeleton of the real GLM-5.3-Flash and DeepSeek-V4-Flash configs, built on the meta device with transformers 5.23.0 and no weight read, gives the MoE layers stated under "Layer range", and the layer count of the config equals the layer count of the skeleton, so the state check before the skeleton uses the right resume key.

## Cost on one H200

Rates are assumptions, set in `modal_giant.py` with `--mbps` and `--layer-seconds`: download 300 MB/s from the hub to the container (a guess; at 100 MB/s the download hours triple) and 30 s per layer for 352 texts of up to 256 tokens (a guess; no giant layer has been run). The table is for pick counts alone. With saliency the routed experts of each layer run twice, so the compute hours are at most twice the column below and the download does not change; `modal_giant.py` prints its estimate with that factor of 2 unless `--no-saliency` is given. H200 at $4.5 per hour. Download bytes for V4-Flash are measured from the shard headers (155.0 GB with embedding and layers, without the head and the MTP layer). The others use the repo sizes from the API, which include the head, MTP and vision tensors and so are a little high.

| Model | Download | Download h at 300 MB/s | Compute h at 30 s/layer | Total h | Cost | Peak decoded layer in bf16 |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | 155 GB | 0.14 | 0.36 | 0.50 | $2.3 | 12.9 GB |
| GLM-5.3-Flash | 328 GB | 0.30 | 0.38 | 0.68 | $3.1 | 14.5 GB |
| Qwen3.5-397B-A17B | 807 GB | 0.75 | 0.50 | 1.25 | $5.6 | 12.9 GB |
| DeepSeek-V4-Pro | 865 GB | 0.80 | 0.51 | 1.31 | $5.9 | 50.7 GB |
| Kimi-K3 | 1561 GB | 1.45 | 0.78 | 2.22 | $10.0 | 59.2 GB |
| Qwen3.8-2.4T-A95B | 4892 GB | 4.53 | 0.77 | 5.30 | $23.8 | 51.5 GB |

At 100 MB/s the totals are 0.8, 1.3, 2.7, 2.9, 5.1 and 14.4 hours ($3.6, $5.8, $12.3, $13.1, $23.0, $64.6).

Memory: the decoded experts of one layer live on the GPU, so the largest layers (Kimi-K3 59 GB, V4-Pro 51 GB, Qwen3.8 52 GB) plus 352 cached hidden states (V4 keeps 4 streams per token, Kimi keeps a growing block residual) fit 141 GB. Qwen3.8 reads one 51 GB fused tensor pair per layer into host memory first (long ranges go out as requests of at most 256 MB and are assembled in one buffer, so the host holds about one copy), and the next layer is prefetched into a second copy, so the function asks for 256 GB of host memory. Dense tensors are decoded and copied into the layer one at a time, so the device holds the layer plus one decoded tensor, not two layers.

The Hub also has `Qwen/Qwen3.8-2.4T-A95B-FP8`, which would halve the download. The plain layout does not decode its scales, so it is not supported.

## Run

```
MODAL_PROFILE=trisanth .venv/bin/modal run modal_giant.py --only zai-org/GLM-5.3-Flash --max-layers 5 --texts 4 --skills math --fresh
MODAL_PROFILE=trisanth .venv/bin/modal run modal_giant.py --only deepseek-ai/DeepSeek-V4-Flash
MODAL_PROFILE=trisanth .venv/bin/modal run modal_giant.py --only moonshotai/Kimi-K3 --no-saliency
MODAL_PROFILE=trisanth .venv/bin/modal run modal_giant.py --estimate-only
```

The first line is the smoke. It takes 5 layers because `--max-layers` counts from layer 0: GLM-5.3-Flash has dense MLPs in layers 0 to 2 and DeepSeek-V4 routes layers 0 to 2 by a token-id table, so 5 layers give two MoE layers with a learned router (layers 3 and 4) for both. With `--max-layers 2` on GLM-5.3-Flash the run stops with an error before any download, because no layer of the range has a router. `--fresh` deletes the saved state of that exact run (`/vol/giant_state/<key>/t<texts>_<skills>`) first. Without it a second identical smoke resumes every layer, runs none, and prints the saved numbers. `--top N` sets how many experts per layer go into `top` and `top_saliency` (default 8). Kimi-K3 runs with `--no-saliency` until its saliency has run on the real class.

It prints the download and hour estimate first. The estimate reads `config.json` as JSON and takes the layout from the zoo row, so it runs with any local transformers version and never runs remote code. `--shard` and `--nshards` split the model list. They cannot split layers, because layer k needs the output of layer k - 1. Output: `/vol/atlas/entries/<org>__<name>.json` (merged into an existing entry, `streamed` field added, `partial` when layers are missing) and `results/giant_<name>.json`.

Saliency in the output. The entry gets `expert_saliency_maps`, `{skill: {layer: [expert ids]}}` from `top_saliency`, merged by layer in the same way as `expert_maps`, which does not change. The entry holds 8 experts per layer by default, and round 19 kept 32 of 128, so the compiler needs the saliency of every expert. That table goes to the volume, `/vol/atlas/saliency/<org>__<name>.json`: `{"general": {layer: [per expert]}, "skills": {skill: {layer: [per expert]}}, "texts": {...}}`, saliency per token for every expert. `giant.top_by_saliency(values, top=n)` ranks a layer of it for any n. It is merged by layer like the entry (`modal_giant.merge_saliency_table`): the layers of a run replace the same layers, the others stay. `texts` has the shape of the table and says for every text set and layer how many texts the numbers come from, so a layer that a 4-text smoke wrote over a full run can be told apart. The results file has a mirror of the merged table under `saliency`. `--no-saliency` runs the pick counts alone; the entry then keeps the saliency maps it already had, the stored table is not touched, and the results file mirrors it (null only when no run with saliency was ever stored).

Output of a run. Per layer: `layer N/T done in S s, read G GB`, and with saliency the lines described under "Log" in the Saliency section. At the end one JSON row per model: `model`, `layers_done`, `moe_layers` (the layer numbers with a router), `seconds`, `gb_read`, `saliency_method` and `saliency_norm` (or `"saliency": "off"`).

Volume. `run_profile` commits the volume in a `finally` after the profile, so the layers saved before a failure reach the volume and a rerun (with `--no-saliency` when the saliency pass failed) continues from them. The image sets `HF_HUB_DISABLE_XET=1`, as `modal_round20.py` does.

## What is verified and what is a guess

Verified:

- The repos, configs, shard counts, byte totals, tensor names and stored dtypes and shapes above (hub API, index files, and shard headers of V4-Flash, GLM-5.3 and Kimi-K3 layer 5).
- Real expert bytes decode to finite weights with a plausible spread (gate std 0.025 for V4-Flash, 0.020 for GLM-5.3, 0.025 for Kimi-K3).
- The exactness tests and the real-class checks listed above. For saliency that means: equal to `slim.collect_moe_stats` on tiny transformers 4.57.1 models through the whole streaming path, and equal to a pair-by-pair reference on single real transformers 5.23.0 MoE blocks of DeepSeek-V4, GLM-5.3 and Qwen3.5-MoE. All on CPU in float32 with random weights. In bfloat16, on CPU, only the self-checks are verified (they pass on the same real blocks and on the toys).
- Kimi-K3 has the norm after the experts: `latent_moe_use_norm` is true in `config.json`, `routed_expert_norm.weight` is in all 92 MoE layers of the index, and the forward of `KimiSparseMoeBlock` applies it to the weighted sum before the up-projection (source read on the Hub, 2026-11-15).
- GLM-5.3-Flash layers 0 to 2 are dense and layer 3 is the first MoE layer; DeepSeek-V4-Flash layers 0 to 2 use the hash router (real configs on the meta device, transformers 5.23.0).
- The router tensor names of V4-Flash, GLM-5.3, Kimi-K3, Qwen3.5 and Qwen3.8, and that no scale tensor sits next to them, from the index files.
- FP4 nibble order and e8m0 scale for DeepSeek, from the reference `convert.py` shipped in the repo.

Guesses and open risks:

- Saliency has not run on a GPU or on real giant weights. In bf16 it ran only on CPU with tiny random weights, and through a whole streamed model of a real transformers 5.x class only for Qwen3.5-MoE. The factor of 2 on compute hours is an upper bound derived from the code, not a timing. The fused experts modules loop over the picked experts, so a layer whose time goes to that loop will come close to the bound.
- Saliency for Kimi-K3 is checked only on toy blocks that copy the forward of `modeling_kimi_linear.py` (ModuleList experts, router returns indices and weights, experts read `routed_expert_down_proj`, RMS norm on the weighted sum, up-projection). The real class has not been run, on any device. Treat Kimi `top_saliency` as unverified and run Kimi with `--no-saliency` until a smoke on the Kimi image passes the self-checks. The choice to divide by the RMS and to stop before the norm weight and the up-projection is a definition, not a measurement: nobody has compared it with a pruning result on Kimi.
- Saliency sums accumulate in float32 per text and float64 across texts. In bf16 the expert outputs are computed again in a batch of another shape, so they can differ from the forward's own values in the last bits. The limit of the self-checks in bf16 (5e-2) is a choice. It is about 9 times the largest difference seen on CPU with tiny random weights (5.6e-3 on the real blocks); real weights on a GPU have not been tried, and a failure there stops the run with the advice to use `--no-saliency`.
- `giant.SALIENCY_STATE_VERSION` protects a rerun only when the version goes up with a change of the saliency code. A change that forgets it is continued from old sums without a message.
- A decoder layer that holds the experts itself, without a MoE block around them, gets no saliency: the meter stops with the advice to run with saliency off. No giant in the zoo has that shape.
- The merge of the saliency table, the `--fresh` delete and the commit in `finally` are tested with a fake volume in a temporary folder. They have not run on a Modal volume. `HF_HUB_DISABLE_XET=1` in the image is copied from `modal_round20.py` and has no test.
- Nothing ran on a GPU or on a real giant layer. The host-memory and device-memory figures above are arithmetic from the tensor sizes, not measurements. Layer time (30 s), download rate (300 MB/s), and whether 352 texts fit the cached-state memory are untested.
- Kimi mxfp4 nibble order (low nibble first, scale `2^(e-127)` per 32) follows the OCP and DeepSeek convention. No Kimi reference code in the repo confirms it. A swapped order would still give plausible magnitudes, so the std check above would not catch it.
- Kimi-K3 image (transformers 4.57.1, `fla-core`, `einops`, sdpa instead of flash_attention_2) has not been built or run. The layer keys match the remote code for all 93 layers, checked with a stub of `fla`. The KDA kernels and sdpa with different q and v head sizes are untested.
- Kimi-K3.1-Pro cannot run until its modeling code exists.
- GLM-5.3 KDA and sparse-attention layers ran only through the transformers reference fallbacks on CPU, not the fast kernels.
- The router and the e_score_correction bias run in bf16 on the H200 (bias in fp32). Expert picks near ties can differ from a model loaded some other way.
- DeepSeek-V4 layers 0 to 2 route by a fixed token-id table, so their expert maps measure the hash of the tokens, not a learned router. Use layers 3 and up for skill selection.
- Models that change the hidden state between layers inside `forward` are not supported. Resuming at a layer after 0 assumes the layer returns a bare tensor, unless the layout sets the output form (Kimi does).
- A giant that needs more than 141 GB for one layer would not fit. None of the listed ones does.
