# Results

Val = validation score (mean ± std over seeds when checked). Test is scored once; its
95% bootstrap CI is in brackets.

## m-brick-kiln

- Published (test): 0.9879, Prithvi-EO-2.0-600M (Prithvi-EO-2.0 paper (arXiv 2412.02732), appendix Tables A1-A4)

| home | version | val | test [95% CI] | GPU min | Bedrock tokens |
|---|---|---|---|---|---|
| gotrain_home | v1 | 0.9897 ± 0.0021 (n=3) | 0.9820 [0.9730, 0.9890] | 35 | 121,133 |
| gotrain_ablation | v1 | 0.9900 ± 0.0017 (n=3) | - | 32 | 92,170 |

## m-chesapeake

- Published (test): 0.7081, DINO ResNet50 (Prithvi-EO-2.0 paper (arXiv 2412.02732), appendix Tables A1-A4)

| home | version | val | test [95% CI] | GPU min | Bedrock tokens |
|---|---|---|---|---|---|
| gotrain_home | v1 | 0.7765 (n=1) | 0.7573 [0.7402, 0.7718] | 106 | 346,459 |

## m-eurosat

- Published (test): 0.9816, MoCo ResNet50 (Prithvi-EO-2.0 paper (arXiv 2412.02732), appendix Tables A1-A4)
- Published (test): 0.9860, TerraFM (fine-tuned) (TerraFM paper (arXiv 2506.06281), fine-tuned result)
- Human baseline (val): 0.9860, hand-trained baseline

| home | version | val | test [95% CI] | GPU min | Bedrock tokens |
|---|---|---|---|---|---|
| gotrain_home | v1 | 0.9856 ± 0.0019 (n=5) | 0.9830 [0.9750, 0.9900] | 17 | 113,894 |

## m-pv4ger-seg

- Published (test): 0.9523, Prithvi-EO-2.0-600M (Prithvi-EO-2.0 paper (arXiv 2412.02732), appendix Tables A1-A4)
- Human baseline (val): 0.9590, hand-trained baseline

| home | version | val | test [95% CI] | GPU min | Bedrock tokens |
|---|---|---|---|---|---|
| gotrain_home | v1 | 0.9582 ± 0.0015 (n=3) | 0.9546 [0.9466, 0.9613] | 95 | 201,653 |
