# Person A: library mining (dissection agent)

An agent that finds labelled parts of open-weight models and validates each label by ablation.

- `dissect/`: the propose -> ablate -> validate -> catalog pipeline. Start with `dissect/README.md`.
  - `agent.py`: the OpenAI agent loop (gpt-6-astra). Tools:
    - weight-space reading (`run_python` in a no-network Modal sandbox with the raw weights)
    - activation profile
    - attribution patching
    - ablate_and_score
    - validate
  - `weight_sandbox.py` + `wsp_helpers.py`: the weight sandbox, modelled on the "Sparks of Weight-Space Programming" harness.
  - `dissect_app.py`: the Modal GPU app with ablation hooks, scoring and the validation evidence.
  - `client.py`: the validation gate and the catalog / .pt index writer.
- `registry/`:
  - `donors.csv`: the donor list
  - `configs/`: each donor's config
  - `tensor_maps/`: safetensors header maps
  - `rangefetch.py`: HTTP range fetch of single tensors
- `catalog.jsonl` + `parts/`: validated parts so far. These are first-pair donor Qwen2.5-Math-1.5B, GSM8K. They are provisional: the gate metric changed after the first candidate (see `dissect/README.md`). They are not merged into the shared top-level catalog yet.
- `experiments.jsonl`: every run, with its config and result.

Secrets: the OpenAI key lives in `.env` (not committed) and the Modal secret `openai-key`.
