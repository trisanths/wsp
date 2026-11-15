# Donor pool notes

Checked 2026-11-14 against Hugging Face configs. Our stack pins
transformers 4.57.1.

## Liquid AI (https://www.liquid.ai/models)

Liquid's LFM models are hybrids: most layers are short gated convolutions
("conv"), and a few are full or sliding attention. That makes them fast and
small for their quality, and they bring layer types no other donor family
has. License: free for commercial use under $10M annual revenue (LFM
License).

| Repo | Type | Params | Layers (conv / attn) | Hidden | Vocab | Needs transformers |
|---|---|---|---|---|---|---|
| LiquidAI/LFM2.5-1.2B-Thinking | lfm2 | 1.2B | 16 (10 / 6) | 2048 | 65536 | 4.57 (works now) |
| LiquidAI/LFM2.5-1.2B-Instruct | lfm2 | 1.2B | 16 | 2048 | 65536 | 4.57 |
| LiquidAI/LFM2-350M-Math | lfm2 | 350M | 16 (10 / 6) | 1024 | 65536 | 4.55+ |
| LiquidAI/LFM2.5-350M, LFM2.5-230M | lfm2 | 350M, 230M | | | | 4.57 |
| LiquidAI/LFM2.5-8B-A1B | lfm2_moe | 8B, 1B active, 32 experts, top 4 | 24 | 2048 | 128000 | 5.x |
| LiquidAI/LFM2.5-24B-A2B | lfm2_moe | 24B, 2B active | | | | 5.x |
| LiquidAI/LFM3-1.7B | lfm3 | 1.7B | 24 (conv, full, sliding) | 2048 | 128000 | 5.x |
| LiquidAI/LFM3-VL-2B, LFM3-Audio-2B, LFM3-Omni-2B | lfm3 multimodal | about 2B | | | | 5.x |

How each fits the language:

- **Taps (donor runs natively):** any LFM model works, because the donor
  runs its own forward pass and the host reads its states. This is the
  natural route for LFM2.5-1.2B-Thinking (reasoning) and LFM2-350M-Math
  (a 350M math specialist, cheap to run beside a host).
- **In-stream grafts:** `mlp_cluster` and `attn_heads` from LFM2 attention
  layers should work. `block` parts that include conv layers need the conv
  state cache (`Lfm2HybridConvCache`), which our part cache does not handle
  yet; uncached blocks may work but are untested.
- **MoE experts:** LFM2.5-8B-A1B and 24B-A2B have 32 small experts per layer,
  good `expert` parts, but they need transformers 5.x.

Open item: LFM2.5 MoE, LFM3, and likely the newest dense families (Qwen3.8,
Gemma 4) need transformers 5.x. Porting wsp from 4.57 to 5.x is a separate
task (meta builds, cache classes, and attention interfaces changed).
