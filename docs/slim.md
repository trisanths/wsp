# Skill removal (slim)

Removing skills is as useful as adding them. If a model only has to write code, its scores on other skills do not matter. The parts that carry those skills can go, and the result is a smaller, faster network that is still good at code.

`wsp/slim.py` holds the mechanisms. `modal_round19.py` measures them. The result is always one network that runs alone. The only training is a short heal.

## Mechanism A: expert removal for MoE models

Rank each layer's experts by how much the skill text uses them. Keep the top N. Delete the rest: the expert list gets shorter and the router loses the rows of the deleted experts. The model's own top-k then picks among the kept experts, capped at N.

Ranking rules, one line each:

- `freq`: how often the router picks the expert on skill text (the same counts as `wsp.atlas.routing_counts` and `expert_skill_map`).
- `saliency`: the sum over skill tokens of the gate weight times the L2 norm of the expert's output (the REAP idea). An expert that is picked often but writes small outputs ranks low.
- `selective`: the log ratio of skill frequency to general frequency. It keeps what the skill uses more than other text does.
- `random`: a seeded random order. Control.
- `general`: frequency on general text. Control: it keeps what general text uses.

N is one count for every layer (`keep=32`), or per layer the fewest experts that take a share of the skill routing (`mass=0.95`). For `saliency` the share is of saliency. For every other rule it is of the routing counts.

Main calls: `collect_moe_stats` (one pass, counts and saliency), `rank_experts`, `choose_keep`, `slim_experts`, and `slim_moe` for all of them at once. `slim_experts` returns a report with parameters total and run per token before and after, and the kept expert ids per layer.

Families are small entries in `FAMILIES`: qwen3_moe, qwen2_moe, olmoe, mixtral, minimax (a softmax router that is an `nn.Linear`), glm4_moe and deepseek_v3 (a top-k router module with one routing group). Transformers 5.x fuses the experts into one module, and that layout raises `NotImplementedError`.

Saving: when every layer keeps the same N, the config gets the new expert count and top-k, and `save_slim` writes a normal Hugging Face directory that `AutoModelForCausalLM.from_pretrained` loads with no wsp code. With a different N per layer, `save_slim` writes the same files plus `wsp_slim.json`, and only `load_slim` rebuilds the model. The manifest field `plain_hf` says which case a directory is.

## Mechanism B: dense slimming

`slim_dense` removes whole layers first, then MLP neurons and attention heads.

- Layers: `wsp.atlas.layer_skill_map` skips each layer once and records the loss increase on skill text and on general text. The default rule `skill_cost` removes the layers with the lowest increase on skill text. Other rules: `specific` (skill increase minus general increase), `influence` (block influence on skill text), `general` (control), `random` (control). The first and last layer always stay. At most 25% of the layers go by default, and an `n_drop` above the cap raises `ValueError`. `greedy_step` re-scores after every step, which accounts for layers that interact.
- Width: neurons and query heads keep the highest mean activation on skill text (`skill`). Other rules: `specific` (Cohen's d, skill against general, from `wsp.prune`), `general`, `random`. Grouped-query attention keeps the same number of query heads per KV head, as in `wsp.prune`.
- Size: `plan_size` turns a parameter target into a layer count and one width fraction. Layers go first, up to the cap. Then neurons and heads thin until the count is closest to the target.

The result is a normal Hugging Face model. The config has the new layer count, head counts, and MLP width.

## Mechanism C: heal

`heal` trains the slimmed model toward the original model's logits. The loss is the KL divergence from the teacher plus the LM loss, on skill text. A LoRA (r16) on the attention linears trains, plus the MLP linears for a dense model. A MoE model also trains its kept routers, in float32. Everything else is frozen. The LoRA merges into the weights at the end, so the model saves as a plain Hugging Face model. Steps, learning rates, and loss weights are parameters.

For MoE, `share_experts=True` makes the student reuse the teacher's expert modules. Only the attention, embeddings, and routers are copied, so teacher and student fit in one GPU. The experts stay frozen, and the test suite checks that the teacher does not change.

## What round 19 measures

Skill: code. Ranking text and heal text are chat-format code solutions from MBPP full train and Magicoder (Python rows). Texts that hold a HumanEval or MBPP test prompt line are dropped, and MBPP rows with a task id in the sanitized test split are dropped.

MoE, `Qwen/Qwen3-30B-A3B-Instruct-2507` (128 experts, top-k 8): the full model; keep N of 64, 32, 16 by saliency; N=32 by freq and by selective; N=32 random (three seeds) and general controls; N=32 and N=16 saliency with a 500-step heal; N=32 random with the same heal (three seeds); and the full model with the same heal, which shows what a code heal adds with nothing removed.

Dense, `Qwen/Qwen2.5-Coder-7B-Instruct`: the full model; slim to about 3.1B and about 1.6B by skill, with and without a 1000-step heal; the same sizes with random removal; one greedy layer choice at 3.1B; the native `Qwen2.5-Coder-3B-Instruct` and `Qwen2.5-Coder-1.5B-Instruct` as the existing-model bar, and the native 3B with the same 1000-step heal.

Every cell scores HumanEval and MBPP (the kept skill) and GSM8K and MMLU (expected to fall; this shows what was removed). Rows hold the scores, parameters total and run per token, decode tokens per second at batch 1, the size of the saved directory, and heal seconds.

Every cell ranks on the same 256 code texts, and the heal texts follow them in a fixed pool, so the ranking does not change with the heal length. The cached ranking statistics are keyed by a hash of those texts. A cell that fails leaves a row with an error, so a rule reads as undecidable instead of silently passing.

Go and stop rules are in the docstring of `modal_round19.py`, and `decide(rows)` computes them. `code_mean` is the mean of the HumanEval and MBPP scores.

1. MoE, no heal: the full model minus N=32 saliency is at most 0.02, and N=32 saliency beats the best of the three random seeds by more than their spread.
2. MoE, heal: the better of the full model and the healed full model minus healed N=32 saliency is at most 0.01, and it beats the best of the three healed random seeds by more than their spread. The healed full model is the reference because the heal data uses the scoring prompt format, so part of a gain is format, not recovery.
3. Dense: the slimmed 3.1B model with the heal beats both the native Coder-3B and the native Coder-3B with the same heal.
4. Size: dense cells land within 3% of the target, and the skill and random cells of one size have the same total parameters.

The standard error of `code_mean` is about 1.7 points, so a gap near a threshold is not a clear result. Rows hold the per-item results of both code skills, and `decide` reports a paired bootstrap 95% interval for the key gaps.

Planned shapes, checked on the real configs without weights: N=32 leaves 8.78B parameters in total and 3.03B run per token (the full model has 30.53B and 3.04B). The 3.1B dense target removes 7 of 28 layers and keeps 40% of the neurons and heads. The 1.6B target keeps about 9% of them, because the embedding and head hold 1.09B parameters that slimming does not touch. That cell is likely to fail and marks where the path ends.

## What is unverified

- Nothing here has run on a GPU. Tests use tiny random models on the CPU. No quality number exists yet.
- Expert removal changes nothing for text that never uses the removed experts, but only for families that renormalize the top-k weights (Qwen3-MoE, Mixtral, MiniMax, GLM, DeepSeek). OLMoE and Qwen2-MoE keep the softmax over all experts, so their kept weights shift a little after removal. The tests check the exact case with Qwen3-MoE.
- Saliency uses the gate weight and output norm on the ranking texts only. With 256 texts, an expert that matters for rare code is under-counted. Whether 256 is enough is not measured.
- The router learning rate for the heal (2e-5) is a guess from the weight scale. A larger step reorders the top-k. It has not been tuned.
- Layer costs from one-shot skipping ignore that layers interact. The greedy cell tests this at one size.
- Width thinning by mean activation is a heuristic. Earlier runs here found that activation-picked neuron clusters did no better than random ones, and that halving a dense model's depth failed (run 15). The random controls exist to catch this.
- The decode speed is the Hugging Face eager loop over experts. It does not show what a fused MoE kernel would give, and a smaller expert list may gain less there than in this loop.
- Qwen3-MoE, the HF save, and the load path are tested at tiny sizes. A 30B save and reload is not tested.
- The GLM adapter is tested on tiny models for no-op removal, saving, and per-layer loading, and the DeepSeek adapter for no-op removal and statistics. MiniMax is listed but has no test. None ran on a real checkpoint.
