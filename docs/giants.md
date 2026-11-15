# Giant open MoE models as component sources

Date of checks: 2026-11-14. Code: `wsp/giant.py`, `modal_giant.py`, `tests/test_giant.py`. Zoo rows: `atlas/zoo.json` (field `stream`).

No machine we can rent holds a giant. Sapien needs two things from one: which experts matter for a skill (the atlas expert map), and the weights of the chosen experts. `wsp/giant.py` gets both by reading the repo with HTTP range requests. It never downloads a whole model.

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

`giant.profile_streaming(source, config, tokenizer, skill_texts, general_texts, ...)` returns `{"maps": {skill: map}, "info": {...}}`. Each map has the format of `wsp.atlas.expert_skill_map`: `layers[layer]` with `skill_freq`, `general_freq`, `log_ratio`, `top`, plus `top_k` and `num_experts`. `atlas.make_entry` and `atlas.catalog_rows` take it unchanged.

Per layer: read the layer's tensors by byte range (merged into few requests), build the layer with the transformers class on the device, decode FP8, FP4 and mxfp4 on the device, write the experts into the layer, run every cached hidden state through it, hook the router, keep the output states, free the layer. The next layer's tensors are fetched in a background thread while the current one runs.

How the layer gets the same inputs as in the full model: the model is built on the meta device with each decoder layer replaced by a stub. The model's own `forward` runs per text and builds the masks, position embeddings and hash-routing inputs. Stubs before the active layer pass their input through. The active stub injects the cached state (for Kimi, the hidden state and the block residual), runs the real layer, stores its output and stops the forward. Models whose forward changes the hidden state between layers are not handled.

Resume: `state_dir` saves router counts after every layer and hidden states every few layers. A rerun with the same texts continues from the last saved states.

### Single experts

`giant.fetch_expert(source, layer, expert_ids, layout=...)` returns `gate`, `up`, `down` (decoded, or stored with their scales when `decode=False`), `router_rows` and `router_bias` for the chosen experts, by byte range. `giant.expert_download_bytes(source, layout, {layer: [ids]})` reports the bytes from the shard headers. Measured on the real repos: one V4-Flash expert is 13.4 MB (fp4 plus scales), one GLM-5.3 expert 25.2 MB, one Kimi-K3 expert 17.6 MB. Eight experts in every MoE layer of V4-Flash are about 4.6 GB, against 155 GB for the model.

## Verification

Exactness (CPU, `tests/test_giant.py`, 59 tests):

- Streamed expert maps equal `atlas.expert_skill_map` on the fully loaded tiny model, exactly (same floats), for qwen3_moe, olmoe, mixtral, deepseek_v2 and deepseek_v3, saved as safetensors. A dense layer (qwen3_moe with `mlp_only_layers`, deepseek first-k-dense) gives no entry, as in atlas.
- Streamed hidden states equal `output_hidden_states` of the full forward after 2 layers (`torch.equal`).
- `fetch_expert` equals direct state-dict reads for per-expert and fused checkpoints, with router rows and bias. `expert_download_bytes` equals the bytes read.
- Toy giants written in the V4, GLM and Kimi checkpoint formats (FP8 with block scales, FP4 with e8m0, mxfp4 as `weight_packed`, hash routing, q/k/v conv concat, a two-state layer with a block residual, extra attributes on layers) match the full model. These toys run in transformers 4.57.1 with custom classes.
- Resume, prefetch and refusal of dense models.

Checks against the real classes and real key names, run in a scratch environment with transformers 5.23.0 (not part of the default test run):

- Real index key names of V4-Flash (43 layers) and V4-Pro (61), GLM-5.3 (45), Kimi-K3 (93) and Qwen3.8 (92) and Qwen3.5-397B (60), mapped through the layouts, give exactly the state-dict keys of the transformers layer, in every layer. Kimi-K3.1-Pro differs in 24 layers (the indexer keys).
- End to end on tiny random models of the real classes: a V4 model with all 43 layer types (compressor, indexer, hash routing, hyper-connections) written as a native checkpoint with FP8 dense and FP4 experts, a GLM-5.3 model with 45 layers written with FP8, and a Qwen3.8 model with fused experts. In each, the streamed expert map equals `expert_skill_map` of the full model with the decoded weights.

## Cost on one H200

Rates are assumptions, set in `modal_giant.py` with `--mbps` and `--layer-seconds`: download 300 MB/s from the hub to the container (a guess; at 100 MB/s the download hours triple) and 30 s per layer for 352 texts of up to 256 tokens (a guess; no giant layer has been run). H200 at $4.5 per hour. Download bytes for V4-Flash are measured from the shard headers (155.0 GB with embedding and layers, without the head and the MTP layer). The others use the repo sizes from the API, which include the head, MTP and vision tensors and so are a little high.

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
MODAL_PROFILE=trisanth .venv/bin/modal run modal_giant.py --only zai-org/GLM-5.3-Flash --max-layers 2 --texts 4 --skills math
MODAL_PROFILE=trisanth .venv/bin/modal run modal_giant.py --only deepseek-ai/DeepSeek-V4-Flash
MODAL_PROFILE=trisanth .venv/bin/modal run modal_giant.py --estimate-only
```

It prints the download and hour estimate first. The estimate reads `config.json` as JSON and takes the layout from the zoo row, so it runs with any local transformers version and never runs remote code. `--shard` and `--nshards` split the model list. They cannot split layers, because layer k needs the output of layer k - 1. Output: `/vol/atlas/entries/<org>__<name>.json` (merged into an existing entry, `streamed` field added, `partial` when layers are missing) and `results/giant_<name>.json`.

## What is verified and what is a guess

Verified:

- The repos, configs, shard counts, byte totals, tensor names and stored dtypes and shapes above (hub API, index files, and shard headers of V4-Flash, GLM-5.3 and Kimi-K3 layer 5).
- Real expert bytes decode to finite weights with a plausible spread (gate std 0.025 for V4-Flash, 0.020 for GLM-5.3, 0.025 for Kimi-K3).
- The exactness tests and the real-class checks listed above.
- FP4 nibble order and e8m0 scale for DeepSeek, from the reference `convert.py` shipped in the repo.

Guesses and open risks:

- Nothing ran on a GPU or on a real giant layer. The host-memory and device-memory figures above are arithmetic from the tensor sizes, not measurements. Layer time (30 s), download rate (300 MB/s), and whether 352 texts fit the cached-state memory are untested.
- Kimi mxfp4 nibble order (low nibble first, scale `2^(e-127)` per 32) follows the OCP and DeepSeek convention. No Kimi reference code in the repo confirms it. A swapped order would still give plausible magnitudes, so the std check above would not catch it.
- Kimi-K3 image (transformers 4.57.1, `fla-core`, `einops`, sdpa instead of flash_attention_2) has not been built or run. The layer keys match the remote code for all 93 layers, checked with a stub of `fla`. The KDA kernels and sdpa with different q and v head sizes are untested.
- Kimi-K3.1-Pro cannot run until its modeling code exists.
- GLM-5.3 KDA and sparse-attention layers ran only through the transformers reference fallbacks on CPU, not the fast kernels.
- The router and the e_score_correction bias run in bf16 on the H200 (bias in fp32). Expert picks near ties can differ from a model loaded some other way.
- DeepSeek-V4 layers 0 to 2 route by a fixed token-id table, so their expert maps measure the hash of the tokens, not a learned router. Use layers 3 and up for skill selection.
- Models that change the hidden state between layers inside `forward` are not supported. Resuming at a layer after 0 assumes the layer returns a bare tensor, unless the layout sets the output form (Kimi does).
- A giant that needs more than 141 GB for one layer would not fit. None of the listed ones does.
