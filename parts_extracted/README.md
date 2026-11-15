# Extracted parts (Person B, for Person A's catalog)

Index files written by `modal_eval.py::extract` (see `wsp_eval/extraction.py`).
Each `.pt` is `{kind, model, layer, indices, signature}`; each part has 3 random
same-layer controls (`*_control_seed{0,1,2}.pt`). `records.jsonl` holds catalog-ready
records; `units` paths are relative to the repo root (on the Modal volume they live
under `/vol/eval/extracted/`).

Method: activation contrast (Cohen's d of |activation|) between GSM8K heal-split text
(train, 256 texts) and WikiText-2 (256 texts), top 3 layers per kind.

| Donor | License | mlp_cluster layers | attn_heads layers |
|---|---|---|---|
| Qwen/Qwen2.5-Math-1.5B | apache-2.0 | 16, 17, 21 | 20, 21, 22 |
| microsoft/phi-4 | mit | 16, 17, 18 | 5, 6, 33 |
| Nanbeige/Nanbeige4.3-33B | apache-2.0 | 33, 34, 36 | 14, 20, 24 |

Contrast effect is a selection statistic, not evidence of function. Ablation checks
on Qwen2.5-Math did not pass as specific (see `wsp_eval/README.md`). Repo ids and
licenses checked against the Hugging Face API (public, ungated, safetensors).
