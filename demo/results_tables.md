### clinic

"I run a clinic. I want a private assistant that answers medical questions as well as Qwen3-30B, on a smaller GPU we can keep on site."

| cell | total params (B) | run per token (B) | MedMCQA val 1000 | MMLU medical test (945) | MMLU general 1000 (no medical) | disk (GB) | source |
|---|---|---|---|---|---|---|---|
| full model | 30.53 | 3.04 | 68.0 | 86.9 | 77.2 | 61.1 | results/round28_rows.json cell full_30b_mcq |
| keep 64 by saliency, no heal | 16.03 | 3.04 | 65.9 | 84.3 | 65.0 | 32.1 | results/round28_rows.json cell clinic_sal64 |
| keep 48 by saliency, no heal | 12.40 | 3.03 | 64.5 | 81.3 | 56.8 | 24.8 | results/round28_rows.json cell clinic_sal48 |
| keep 32 by saliency, 500-step heal | 8.78 | 3.03 | 64.0 | 79.0 | 52.1 | 17.6 | results/round28_rows.json cell clinic_sal32_heal500 |
| keep 48 random (seed 0), no heal | 12.40 | 3.03 | 31.1 | 21.7 | 23.8 | 24.8 | results/round28_rows.json cell clinic_random48_s0 |

### accounting

"We do bookkeeping and tax prep. I need a model that's great at word-problem arithmetic and finance math, and cheap to run."

| cell | total params (B) | run per token (B) | GSM8K test 500 | MATH-500 | MMLU accounting + statistics test (498) | MMLU general 1000 (no acct/stats) | disk (GB) | source |
|---|---|---|---|---|---|---|---|---|
| full_30b | missing | |  |  |  |  | | |
| keep 64 by saliency, no heal | 16.03 | 3.04 | 91.4 | 42.4 | 64.1 | 53.6 | 32.1 | results/round28_rows.json cell accounting_sal64 |
| keep 48 by saliency, no heal | 12.40 | 3.03 | 90.6 | 48.2 | 56.2 | 43.6 | 24.8 | results/round28_rows.json cell accounting_sal48 |
| accounting_sal32_heal500 | missing | |  |  |  |  | | |
| keep 48 random (seed 0), no heal | 12.40 | 3.03 | 0.0 | 0.0 | 20.1 | 23.8 | 24.8 | results/round28_rows.json cell accounting_random48_s0 |

### law_firm

"We're a small law firm. We want a model that knows law well for first-pass research and runs on our own hardware."

| cell | total params (B) | run per token (B) | MMLU law test (1763) | MMLU general 1000 (no law) | disk (GB) | source |
|---|---|---|---|---|---|---|
| full model | 30.53 | 3.04 | 62.7 | 80.3 | 61.1 | results/round28_rows.json cell full_30b_mcq |
| keep 64 by saliency, no heal | 16.03 | 3.04 | 61.8 | 60.3 | 32.1 | results/round28_rows.json cell law_firm_sal64 |
| keep 48 by saliency, no heal | 12.40 | 3.03 | 60.4 | 55.7 | 24.8 | results/round28_rows.json cell law_firm_sal48 |
| keep 32 by saliency, 500-step heal | 8.78 | 3.03 | 50.5 | 46.0 | 17.6 | results/round28_rows.json cell law_firm_sal32_heal500 |
| keep 48 random (seed 0), no heal | 12.40 | 3.03 | 24.4 | 24.4 | 24.8 | results/round28_rows.json cell law_firm_random48_s0 |
