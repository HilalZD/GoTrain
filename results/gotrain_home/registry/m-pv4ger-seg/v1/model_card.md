# m-pv4ger-seg v1

- Run: `m-pv4ger-seg-20261031-193048-992b`
- Model: single run
- Validation miou: **0.9582 ± 0.0015** (mean of 3 seeds); single-run score 0.9577
- Stopped by its time budget before max_epochs (the curve may still rise)
- Training time: 9.9 min, 13 epochs
- Lineage (newest first): `m-pv4ger-seg-20261031-193048-992b`
- Git commit: `65537a79e384bbdbe0a3496ba25fc7a077827584`
- Registered: 2026-10-31T20:22:37.313700+00:00

## Data

- task_type: segmentation
- sensor: aerial-rgb
- n_bands: 3
- n_classes: 2
- image_size: [320, 320]
- split_sizes: {'train': 3000, 'val': 403}

## Config

```json
{
  "task": "m-pv4ger-seg",
  "backbone": "convnext_tiny",
  "weights": "imagenet",
  "bands": "all",
  "img_size": 320,
  "batch_size": 16,
  "lr": 0.0002,
  "weight_decay": 0.01,
  "max_epochs": 30,
  "loss": "ce",
  "augment": [
    "flip",
    "rot90"
  ],
  "train_subset_frac": 1.0,
  "seed": 0,
  "reason": "Per VISUAL OBSERVATION (fragmented masks, missed dark panels) and HYPOTHESIS (loss matters): test ce on convnext at same 10 min budget"
}
```

## Notes

U-Net convnext_tiny (imagenet), ce loss, lr 2e-4, wd 0.01, flip+rot90, 320px, bs16, 10 min budget (~13 epochs). Chosen by mean validation mIoU over 3 seeds: 0.9582 ± 0.0015 (single runs 0.9577/0.9570/0.9598). Ensembles/TTA gained at most +0.001 (3-member+TTA 0.9608), which is within seed noise, so the simplest model was kept. Doubling training time to 20 min gave no gain (0.9577). Human baseline (0.959 val): effectively matched, mean 0.9582 is 0.0008 below, within noise. Best published reference 0.9523 (test) is exceeded. Residual errors: dark or low-contrast panels missed, skylights confused with panels.

## Test result

- Test miou: **0.9546**
- 95% CI: [0.9466, 0.9613] (95% bootstrap CI)
- Scored: 2026-11-01T09:47:09.105754+00:00
