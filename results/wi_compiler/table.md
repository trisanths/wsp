| arm | seed | MMLU held-out 2000 | ARC-C test | OBQA test | MMLU dev 1000 | params/token | trainable | GPU h | tok/s | source |
|---|---|---|---|---|---|---|---|---|---|---|
| base (wi_xfam scoring) | 0 | 44.6 | 55.3 | 60.0 | 48.0 | 1235814400 | 0 | 0.005 | 103.21 | results/wi_xfam/rows/reference_0_final.json |
| base (compiler scoring) | 1 | 44.6 | 55.3 | 60.0 | 48.0 | 1235814400 | 0 |  | 94.55 | results/wi_compiler/xfam_knowledge_report.json |
| graft, compiler path | 1 | 58.7 | 84.5 | 88.6 | 58.3 | 5213011330 | 26486658 | 0.681 | 30.18 | results/wi_compiler/xfam_knowledge_report.json |
| graft, wi_xfam driver | 1 | 59.5 | 85.2 | 87.0 | 57.2 | 5213011330 | 26486658 | 0.659 | 29.47 | results/wi_xfam/rows/graft_lora_1_final.json |
| identity control, compiler path | 1 | 50.4 | 70.7 | 76.6 | 50.7 | 1251028866 | 26486658 | 0.31 |  | results/wi_compiler/xfam_knowledge_report.json |
| scrambled-donor control, compiler path | 1 | not run | | | | | | | | the compile lost its GPU workspace during this control |
| identity control, wi_xfam driver | 1 | 50.9 | 69.9 | 77.4 | 52.2 | 1251028866 | 26486658 | 0.319 | 53.05 | results/wi_xfam/rows/identity_l_1_final.json |
| scrambled-donor control, wi_xfam driver | 1 | 50.3 | 70.1 | 78.6 | 52.6 | 5213011330 | 26486658 | 0.635 | 32.28 | results/wi_xfam/rows/scrambled_l_1_final.json |
| LoRA matched to graft, wi_xfam driver | 1 | 50.5 | 68.9 | 77.0 | 50.7 | 1235814400 | 26066944 | 0.307 | 62.16 | results/wi_xfam/rows/lora_match_l_1_final.json |
| LoRA rank 128, wi_xfam driver | 1 | 50.6 | 69.7 | 79.6 | 53.8 | 1235814400 | 90177536 | 0.285 | 72.31 | results/wi_xfam/rows/lora_big_1_final.json |
| full fine-tune, wi_xfam driver | 1 | 51.1 | 69.4 | 77.8 | 53.1 | 1235814400 | 1235814400 | 0.319 | 116.53 | results/wi_xfam/rows/fullft_1_final.json |

Compiler graft against the wi_xfam graft of seed 1 on MMLU held-out 2000: -0.8 points (limit 1.5). One seed each.
