# Benchmark research agent: quality check, 2026-11-15

The agent in `studio/server/bench_research.py` reads a request, searches the web with Exa, and proposes benchmarks with reference scores from published pages. This note grades 8 requests before and after a set of fixes. Every number below was checked by opening the source page.

## Method

- `scripts/eval_bench_research.py` runs the pipeline directly, with no server, and writes one JSON record per request to `outputs/bench_research_eval/<tag>/<n>.json`. A record has the steps, proposals, sources, timings, token use, the raw LLM replies, the first search results and the page text the score check saw. `outputs/` is git ignored. `--offline` runs the same code with a fake search and a fake LLM.
- Eight real runs before the fixes (`before`), eight real runs with the new prompts and a first version of the checker (`after`), and four real runs of the final code on requests 1, 2, 5 and 6 (`live2`). That is 20 real research runs in all.
- Grading: for each sourced score I opened the page (Exa get-contents) and checked four things: the benchmark is real and fits the request, the model exists, the number is on the page for that benchmark and that model (not another column, metric or model), and the size fits the limit. A score is "fully right" only when all four hold. "Number right" means only the third.
- The before sample is 59 of the 73 sourced scores. I did not pick at random. I made sure every source site and every table style was in it. The 14 not opened were mostly more rows of tables I had already opened for another row. The after sample is all 24 sourced scores.
- The after column comes from replaying the recorded LLM replies and page text through the final checker (`scripts/replay_bench_research.py`). This keeps the LLM and the search the same and changes only the code under test. The replay of the four `live2` requests gave the same counts as the live runs except one dropped duplicate. For requests 0, 3, 4 and 7 the first search pages were not stored by the first record, so I fetched the cited pages again in full. A full page is a little easier to match than the short highlights the live run sees, so those four after numbers may be a little high.

## Results

Request numbers: 0 legal, 1 coding assistant, 2 nurses, 3 math tutor, 4 telecom support, 5 English-Spanish, 6 coding-only shrink of Qwen2.5-Coder-7B, 7 general reasoning assistant.

| # | Benchmarks picked (before) | Sourced before | Checked | Number right | Fully right | Sourced after | Right after |
|---|---|---|---|---|---|---|---|
| 0 | MMLU-Pro law, LegalBench, LawBench, two Bangladesh legal QA sets, IFEval | 8 | 6 | 4 | 3 | 1 | 1 of 1 |
| 1 | HumanEval, IFEval, MMLU-Pro CS, GSM8K, BBEH | 8 | 7 | 6 | 6 | 9 | 9 of 9 |
| 2 | MMLU-Pro health, MMLU, IFEval, MMLU-Pro biology, chemistry, psychology | 10 | 8 | 3 | 3 | 3 | 3 of 3 |
| 3 | GSM8K, IFEval, BBEH, MMLU-Pro math, MMLU | 5 | 4 | 1 | 1 | 6 | 6 of 6 |
| 4 | TelBench, IFEval, MMLU, BBEH, MMLU-Pro business | 8 | 7 | 6 | 1 | 1 | 1 of 1 |
| 5 | FLORES, WMT, Tatoeba, NTREX, IFEval | 8 | 6 | 4 | 2 | 0 | none to check |
| 6 | HumanEval, IFEval, MMLU-Pro CS, MultiPL-E ("MBPP style"), LiveCodeBench | 11 | 10 | 6 | 6 | 4 | 4 of 4 |
| 7 | BBEH, GSM8K, IFEval, MMLU, HumanEval, MMLU-Pro math | 15 | 11 | 9 | 7 | 0 | none to check |
| all | | 73 | 59 | 39 (66%) | 29 (49%) | 24 | 24 of 24 (100%) |

Benchmarks picked after the fixes: 0 LegalBench, MMLU-Pro law, CaseHOLD, IFEval, MMLU. 1 HumanEval, MBPP, BigCodeBench, IFEval, MMLU. 2 MedQA, MedMCQA, PubMedQA, MMLU-Pro health, IFEval, MMLU. 3 GSM8K, IFEval, BBH, SVAMP, ASDiv, MMLU. 4 tau-bench telecom, MultiWOZ, IFEval, Bitext customer support, MMLU. 5 FLORES-200, WMT, Tatoeba, NTREX, IFEval. 6 HumanEval, EvalPlus, BigCodeBench, IFEval, MMLU. 7 BBEH mini, GSM8K, MMLU-Pro math, MMLU-Pro philosophy, IFEval.

Cost and time, mean per run: 13.2 s before and 13.2 s after. LLM input tokens 19.6k before and 21.4k after (+9%), output 1.5k before and 1.6k after. The extra input is the longer prompts.

### What improved

- Precision of sourced scores: 49% fully right before, 24 of 24 after. The wrong scores were dropped by the checks or never extracted.
- Misses fixed: MedQA, MedMCQA and PubMedQA now appear for the nurse request (request 2). MBPP now appears for the coding assistant (request 1). SVAMP and ASDiv appear for the math tutor, and tau-bench telecom and MultiWOZ for the telecom request. Request 2 now has three real small-model scores (BioMedLM, 2.7B, on MedQA, MedMCQA and PubMedQA) from a paper and a Stanford CRFM page.
- Each proposal has a confidence note, a metric, and a source kind on each score. Duplicates are gone (before, request 7 listed Llama-3.2-1B four times and request 1 listed Qwen2.5-Coder-1.5B-Instruct twice with 70.7 and 54.5).

### What did not improve

- The number of sourced scores fell from 73 to 24. The checks dropped 30 scores in the eight final runs: 10 for models far above the limit, 11 closed or huge models, 5 changed copies of a model, and 4 where the benchmark or the model was not tied to the number. Another 14 scores stayed as estimates because the quote was not on the page or had no number. The rest of the fall is the LLM returning fewer scores once the prompt told it to skip other columns and big models. I cannot say how many of the 29 right "before" scores the final checker would also accept, because the before records have no quotes. Scores that are right but laid out so a quote cannot prove them (a number whose table has no header row, or a benchmark named only in a figure caption) are now estimates. That is the price of the strict rule.
- Request 5 (translation) and request 7 (reasoning puzzles) have no sourced score at all after the fixes. In request 5 the pages the search found were for 1.7B to 4B models or closed models, and the BLEU tables of the small Helsinki models did not pass the benchmark check. In request 7 the BBEH leaderboard lists only large or closed models. An honest empty result is better than the wrong ones, but the owner gets no number to anchor a target.
- Kind: for request 4 ("follows strict response formats") IFEval is still a guard, and the request asks for it as a target. Request 7 still proposes MMLU-Pro math and philosophy as targets for a reasoning-puzzle assistant. MBPP is still missing for request 6.
- The checker cannot repair a page that misprints its own header. One arXiv table gave a column header "MBPP" to the second number in a row whose real header is HumanEval pass@2. The check passed because the page said so, and the number happened to equal the true MBPP pass@1.
- Suggested targets were not changed by the code. They look plausible for the size on the after runs (GSM8K 0.8 for a 1.5B math model, HumanEval 0.5, MMLU-Pro law 0.35). Before, GSM8K was 0.5, which is low for 1.5B (Qwen2.5-1.5B scores about 0.6).

## Failure patterns, with a real example each

1. A number from a neighbor column or row. `budecosystem/code-millenials-1b` HumanEval 0.8048 (request 1): the page says 80.48 for Code Millenials 34B and 51.82 for 1B. Thomson-1.0-Small LegalBench 0.809 (request 0): 80.9 is the Snowdon column, Thomson is 79.9. MedGemma 1.5 4B "MMLU-Pro biology" 0.735 (request 2): 73.5 is the US-DermMCQA column. LegalOne-1.7B 0.3963 (request 0): 39.63 is the Qwen3-4B row, LegalOne is 41.24.
2. A near name, not the benchmark. MedGemma "MMLU-Pro health" 0.696 (request 2) is the "MMLU Med" row. Qwen2-0.5B "MMLU-Pro biology" 0.147 and InternVL3_5 "MMLU-Pro chemistry" 0.3834 are the overall MMLU-Pro scores (chemistry is 34.19). Qwen2.5-Coder-3B "LiveCodeBench" 0.411 and "MMLU-Pro computer science" 0.652 (request 6) are the BigCodeBench Full and MBPP 3-shot columns of one arXiv table. The same run also listed the 3B model above the 7B model on LiveCodeBench, which no table supports.
3. A changed copy of the model. Qwen2.5-1.5B GSM8K 0.7142 (request 3) is after SFT plus GRPO on a student's repo. Fast-dLLM-v2-1.5B 0.5929 is the "+ Sparse-dLLM" row. Llama-3.2-1B GSM8K 0.3963 (request 7) is an INT8 or FP8 row. DeepseekCoder-1.3B + DR-CoT 0.641 is a prompting method. Qwen2.5-Coder-7B HumanEval 0.909 (request 6) is the "Unit Tests" selection column.
4. Models far from the size. Mistral-Large, Claude 3.5 Sonnet and Llama-3.1-405B on TelBench (request 4), SaulLM-54B on MMLU-Pro law (request 0), o3-mini and DeepSeek R1 on BBEH (request 7), Gemma3-4B on a 1B translation request (request 5).
5. Mixed metrics. TelBench scores were F1, ROUGE-L and accuracy of different sub-tasks listed as one benchmark (request 4). Helsinki opus-mt-tc-big-en-es 0.572 (request 5) is a Tatoeba BLEU of 57.2 shown as a 0 to 1 score on a FLORES row (FLORES BLEU is 28.5).
6. Wrong language or wrong task on a good looking page. NLLB-200 0.4926 (request 5) is a Turkish row on the NTREX-128 page, not English to Spanish. Qwen2.5-7B-Instruct "MMLU" 0.4855 (request 4) is the Open LLM Leaderboard `acc_norm` aggregate, not MMLU.
7. Duplicates and conflicts. Llama-3.2-1B four times with different scores (request 7). Qwen2.5-Coder-1.5B-Instruct 70.7 (model card) and 54.5 (a paper table with its own harness) in the same list (request 1).
8. Weak fit and misses. Two "Bangladesh legal QA" sets from one aggregator were proposed as targets for a general legal assistant (request 0). MedQA, PubMedQA and MedMCQA were missing for nurses and four MMLU-Pro subjects were proposed instead (request 2). HumanEval as a target for a reasoning-puzzle assistant (request 7).

## What changed in the code

All in `studio/server/bench_research.py`; fields are documented in `studio/CONTRACT.md`.

- The synthesis and extraction prompts ask for `metric`, `label` and an exact `quote` with the model name before the number, forbid other columns, changed copies and far larger models, and list standard benchmarks by area. The first pass now uses the same checker as the second pass; before, it only checked that the number was somewhere on the page.
- `check_evidence` and `table_info`: the quote must be on the page (a quote may skip up to three cells). The model must come before the number, or head the number's column, or be the page's own model card. The benchmark must be named in the column header, the row, the heading, the page title or the words just before the number (the words above a table count only when its header names no benchmark). A change word between model and number or in the header rejects the score. In running text another score between the model and the number rejects it.
- `model_span` and `name_goes_on`: Qwen2.5-1.5B does not match Qwen2.5-1.5B-Instruct or Qwen3-8B.
- `size_problem`: a model above 1.5 times the size limit, or a closed or very large model with no size, is dropped unless the request names it.
- `add_score`: near-identical models are kept once and the better source wins. `source_kind`, `by_source`: papers and model cards are searched and read first, aggregators last.
- `finish_scores` and `confidence_for`: metric per proposal, `comparable: false` for a different metric, and a confidence note in plain words.
- Plan queries always include one that asks for small-model results and the size.

## Not done

- A second search for primary sources per benchmark. The score search still runs one query per benchmark and sorts the pages it gets back.
- Making a format-following request pick IFEval as a target, and adding MBPP when HumanEval is chosen. Both are LLM choices and I had no real runs left to test a prompt change.
- Converting BLEU-style numbers. A BLEU value is stored as the page number divided by 100 with `metric: "BLEU"`; the web app should show the metric next to it.

## Reproduce

```
.venv/bin/python scripts/eval_bench_research.py --tag mytag            # 8 requests, real search and LLM
.venv/bin/python scripts/eval_bench_research.py --offline --only 0     # no network
.venv/bin/python scripts/replay_bench_research.py --tag mytag --out mytag_replay   # new checker, same replies
.venv/bin/python -m pytest tests/test_studio_research.py               # network is blocked in these tests
```
