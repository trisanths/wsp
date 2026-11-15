# Write-in: languages (2026-11-15)

Status: the graft, the scrambled-donor control, the base, the donor, and three native models are scored. The
identity control, the two LoRA arms, and the full fine-tune were still running when this was written; their rows
will be added. No verifier has checked these numbers yet.

## The request

"My assistant runs on Llama-3.2-1B. It needs to understand Arabic, Japanese, Chinese, Korean and Hindi as well as it
understands English. Add those languages from a model that already knows them."

Request file: `demo/requests/language.json`. Meta lists eight supported languages for Llama-3.2, and Arabic,
Japanese, Chinese and Korean are not among them. Qwen2.5-7B-Instruct covers all five.

## What was built

One network: Llama-3.2-1B-Instruct with a frozen slice of Qwen2.5-7B-Instruct (layers 4 to 20) read after host layer
7 through a trained encoder, decoder and per-token gate, plus a rank-16 LoRA on the host (26.5M trainable
parameters). The recipe is the one in `docs/write_in_xfam.md`: 600 alignment steps, then 4000 task steps at batch 16
with distillation from the donor's answer distribution, last step scored. Driver: `modal_wi_lang.py`. Rows:
`results/wi_lang/rows/`.

Training pool, the same for every arm (78,457 four-option and three-option items, `results/wi_lang/prep_report.json`):

- MMLU translated into the five target languages (openai/MMMLU), about 9,400 questions per language. The 1,000
  held-out questions and the 200 dev questions are removed in every language, together with their English originals.
- XNLI train recast as three-option items, 4,000 each in Arabic, Hindi and Chinese.
- 19,973 English items from the xfam pool (MMLU auxiliary_train, CommonsenseQA, OpenBookQA, QASC, LogiQA train).

No Belebele item in any language is in the pool or used for any choice. Spanish, French, Bengali and Swahili get no
training text at all.

## Results

Belebele reading comprehension, test split, 900 items per language, accuracy in percent. Graft is the mean of seeds
1, 2, 3; scrambled is the mean of seeds 1, 2; the other rows are one run with no training.

| Model | Params per token | Seeds | English | Arabic | Japanese | Chinese | Korean | Hindi |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B base | 1.24B |  | 58.7 | 42.8 | 45.8 | 54.4 | 45.7 | 38.1 |
| Graft of Qwen2.5-7B layers 4 to 20 | 5.21B | 3 | **89.7** | **75.7** | **76.0** | **84.2** | **77.3** | **67.3** |
| Same graft, donor weights scrambled | 5.21B | 2 | 77.4 | 55.7 | 57.4 | 68.3 | 57.6 | 50.0 |
| Native Qwen2.5-1.5B-Instruct | 1.54B |  | 82.4 | 68.4 | 68.6 | 79.4 | 70.8 | 48.9 |
| Native Llama-3.2-3B-Instruct | 3.21B |  | 86.2 | 69.8 | 70.4 | 78.8 | 70.2 | 63.0 |
| Native Qwen2.5-3B-Instruct | 3.09B |  | 90.4 | 81.9 | 79.7 | 86.6 | 81.3 | 61.7 |
| Qwen2.5-7B-Instruct donor alone | 7.62B |  | 92.4 | 86.0 | 84.9 | 89.6 | 85.7 | 69.8 |

Graft seeds: Arabic 73.4, 75.6, 78.1; Japanese 75.0, 75.3, 77.8; Chinese 83.8, 85.0, 83.9; Korean 76.6, 77.6, 77.7;
Hindi 66.6, 68.6, 66.8; English 88.7, 90.3, 90.1.

Languages with no training text (same Belebele test, 900 items each):

| Model | Params per token | Seeds | Spanish | French | Bengali | Swahili |
|---|---|---|---|---|---|---|
| Llama-3.2-1B base | 1.24B |  | 51.2 | 52.2 | 34.0 | 36.0 |
| Graft of Qwen2.5-7B layers 4 to 20 | 5.21B | 3 | **85.0** | **86.7** | **54.9** | **61.2** |
| Same graft, donor weights scrambled | 5.21B | 2 | 65.8 | 68.8 | 38.8 | 45.2 |
| Native Qwen2.5-1.5B-Instruct | 1.54B |  | 75.0 | 76.0 | 45.4 | 36.9 |
| Native Llama-3.2-3B-Instruct | 3.21B |  | 79.6 | 81.6 | 60.1 | 56.8 |
| Native Qwen2.5-3B-Instruct | 3.09B |  | 84.8 | 85.3 | 61.3 | 42.4 |
| Qwen2.5-7B-Instruct donor alone | 7.62B |  | 88.3 | 89.9 | 69.1 | 42.8 |

Knowledge in each language: MMMLU held-out questions (1,000 per language, never trained on in any language) and
English MMLU held-out 2,000:

| Model | Params per token | Seeds | English MMLU held-out | Arabic | Japanese | Chinese | Korean | Hindi |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B base | 1.24B |  | 44.6 | 33.3 | 34.2 | 37.8 | 34.6 | 30.9 |
| Graft of Qwen2.5-7B layers 4 to 20 | 5.21B | 3 | **59.8** | **44.4** | **48.8** | **50.9** | **45.3** | **43.1** |
| Same graft, donor weights scrambled | 5.21B | 2 | 51.3 | 36.0 | 38.7 | 41.2 | 36.8 | 36.0 |
| Native Qwen2.5-1.5B-Instruct | 1.54B |  | 60.2 | 39.4 | 47.2 | 53.8 | 44.3 | 33.5 |
| Native Llama-3.2-3B-Instruct | 3.21B |  | 57.1 | 40.1 | 42.6 | 47.7 | 41.6 | 40.3 |
| Native Qwen2.5-3B-Instruct | 3.09B |  | 65.8 | 48.0 | 51.5 | 60.2 | 49.0 | 40.2 |
| Qwen2.5-7B-Instruct donor alone | 7.62B |  | 70.7 | 51.3 | 59.6 | 66.6 | 57.4 | 44.1 |

## Reading the tables

- The graft adds 29 to 33 points of Belebele accuracy in each of the five target languages and 31 points in English.
- The scrambled-donor control has the same connector, the same LoRA, the same data and the same steps, with the
  donor slice replaced by weights of the same spectrum and no knowledge. It gains 12 to 14 points over the base,
  which is what fine-tuning through the connector gives. The graft is 17 to 20 points above it, so the rest comes
  from what the donor slice knows.
- With the graft switched off at test time the trained network falls back to the base: Arabic Belebele 46.0, 41.6
  and 39.4 for the three seeds (base 42.8).
- Against native models: the graft is above Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct in all six languages.
  Qwen2.5-3B-Instruct is 2 to 6 points above the graft in Arabic, Japanese, Chinese and Korean and runs fewer
  parameters per token (3.09B against 5.21B); the graft is above it in Hindi (67.3 against 61.7).
- Languages with no training text rise as well. Swahili reaches 61.2, above the donor alone (42.8) and every native
  model scored here. One explanation, not tested: the host already has some Swahili and the graft adds reading
  comprehension skill that carries across languages.
- English does not regress: English MMLU held-out is 59.8 against 44.6 for the base.

## Costs

- The graft runs 5.21B parameters per token (1.24B host, 3.96B frozen donor slice, the connector). Decode was 32.5
  tokens per second at batch 1 in the hook-based path (one measurement, seed 1).
- One graft cell took 0.68 to 0.75 GPU hours on an H200 and read 8.5M training tokens.
- All runs used the Modal workspaces hanavm and rexheng-24542.

## Limits

- The fine-tuning arms (LoRA matched, LoRA rank 128, full fine-tune) and the identity control are not in yet, so
  "beats fine-tuning" is not shown here; the scrambled control is the only trained comparison so far.
- Native model rows are one run each, with no fine-tuning on the pool.
- Scores are answer-letter accuracy. Generation quality in each language was not measured.
- About 1 to 2 percent of the Chinese and Korean MMMLU rows carry an answer letter that differs from the English
  original, so the question-index split is slightly noisy for those two languages.
