# Serving and speed (modal_serve.py)

Run 19 shows that skill removal keeps code quality: Qwen3-30B-A3B-Instruct-2507 with 64 of 128 experts per layer scores HumanEval 87.8 and MBPP 79.0 (full model 91.5 and 84.4). Its decode speed was measured with the Hugging Face MoE loop, about 10 tokens per second, so it supports no speed claim. `modal_serve.py` runs each model in vLLM on one H200 and measures speed and HumanEval together.

## What it measures

One fixed set of 64 prompts: the first 64 users of the `wsp.healdata` "code" pool (seed 0), cut to 1500 characters, rendered with `wsp.skills_eval.render_prompt`. Greedy, 256 new tokens, every sequence runs the full 256 (ignore_eos).

- `decode_tps`: tokens per second at batch 1 and batch 32. Time of the 256-token run minus time of a 1-token run of the same prompts, so prefill does not count.
- `prefill_tps`: all 64 prompts in one call, 1 new token.
- `ttft_ms_median`, `ttft_ms_p90`: wall time of a 1-token request at batch 1.
- `gpu_peak_gib`, `gpu_after_load_gib`: device memory by `nvidia-smi`, polled every 0.5 s. vLLM reserves 90% of the GPU for its KV pool, so the peak is about the same for every cell. Use `weights_gib` and `gpu_after_load_gib` to compare model size.
- HumanEval: 164 items, the prompt (`user_text`, `render_prompt`), grading (`grade_outputs`, which uses `extract_code` and the sandbox) of `wsp.skills_eval`, greedy, 512 new tokens, as in Run 19. No code is copied.

## Cells

| Cell | Model |
|---|---|
| full_30b | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| slim_64 | 64 experts per layer, code saliency, no heal |
| slim_32_heal | 32 experts per layer, code saliency, 500-step heal |
| coder_7b | Qwen/Qwen2.5-Coder-7B-Instruct |
| coder_3b | Qwen/Qwen2.5-Coder-3B-Instruct |

## Version

vLLM 0.32.0 (released 2026-10-18). Its registry maps `Qwen3MoeForCausalLM` and `Qwen2ForCausalLM`, and its supported-models page lists Qwen3-30B-A3B. It pins torch 2.13.0 and transformers >=5.16.1,<5.20.0, and pip brings both. The image has no torch or transformers pin of its own. The other packages are the list of `modal_round20.py` (zstandard, datasets, safetensors, and so on), which `wsp.healdata` and `wsp.skills_eval` need.

Slimmed directories come from transformers 4.57.1 (`save_slim`), with one tensor set per expert. vLLM reads the safetensors files itself and uses the config's `num_experts`, so a plain-HF directory loads. The tokenizer always comes from the base Hub id.

## Slimmed models are not on the volume yet

`modal_round19.py` does not keep its slimmed models. `disk_size` saves each one to `/tmp`, reads its size, and deletes it. Only the ranking statistics persist (`/vol/slim/round19`). The slim cells read `/vol/slim_models/round19/moe_saliency_64` and `/vol/slim_models/round19/moe_saliency_32_heal500`. Until a script writes those directories with `wsp.slim.save_slim` (plain_hf true), the script writes a skipped row with a message and runs the other cells. Pass `--slim-root` to use another directory.

Each Modal workspace (trisanth, trisanthsrini, ztrisanth) has its own `wsp-models` volume. A cell path exists only in the workspace that saved it. A CPU function checks the paths first and starts no GPU for a missing one.

## Run it

Smoke first. It runs `coder_3b` with 8 prompts at batch 1 and 8, no HumanEval, and writes `results/serve_smoke_coder_3b.json`:

    modal run --detach modal_serve.py --smoke

Then:

    modal run --detach modal_serve.py --profile-volume trisanth
    modal run --detach modal_serve.py --cells full_30b,coder_7b --profile-volume ztrisanth

`--profile-volume` is a label for the workspace of the active Modal profile. It goes into the rows and into the missing-path message. Rows go to `/vol/results/serve/<cell>.json` (written by the GPU function) and to `results/serve_<cell>.json`, after each cell. A failed cell leaves a row with `error`.

## Read the numbers

One run per cell, one seed (greedy has none). Decode speed at batch 1 is the number that matters for a single user. A claim of speed needs the HumanEval score of the same row.
