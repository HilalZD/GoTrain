# m-eurosat v1

- Run: `m-eurosat-20261031-180046-56a0`
- Model: single run
- Validation accuracy: **0.9856 ± 0.0019** (mean of 5 seeds); single-run score 0.9850
- Training time: 1.9 min, 30 epochs
- Lineage (newest first): `m-eurosat-20261031-180046-56a0` <- `m-eurosat-20261031-175546-2f56`
- Git commit: `65537a79e384bbdbe0a3496ba25fc7a077827584`
- Registered: 2026-10-31T18:11:02.057975+00:00

## Data

- task_type: classification
- sensor: sentinel-2
- n_bands: 13
- n_classes: 10
- image_size: [64, 64]
- split_sizes: {'train': 2000, 'val': 1000}

## Config

```json
{
  "task": "m-eurosat",
  "backbone": "resnet50",
  "weights": "ssl4eo_s2",
  "bands": "all",
  "img_size": 128,
  "batch_size": 64,
  "lr": 0.0003,
  "weight_decay": 0.01,
  "max_epochs": 30,
  "loss": "ce",
  "augment": [
    "flip",
    "rot90"
  ],
  "train_subset_frac": 1.0,
  "seed": 0,
  "reason": "Halving rung 2 of m-eurosat-20261031-175546-2f56 at 3.5 min. Original reason: upsample to 128 to give pretrained filters more spatial resolution"
}
```

## Notes

Chosen by mean validation score: ResNet50 with ssl4eo_s2 weights, all 13 bands, img_size 128 (upsampled from 64), ce loss, lr 3e-4, wd 0.01, flip+rot90, 30 epochs, about 1.9 min per run. Seed group of 5: mean 0.9856 ± 0.0019 (scores 0.985, 0.988, 0.983, 0.987, 0.985). This matches the human baseline of 0.986 on validation, and it is within noise of the best published test score (0.986). In the one halving screen, 128px beat 64px: the 64px variants scored 0.973–0.979, ImageNet weights scored 0.946, and resnet18 scored 0.978. Ensemble and TTA were evaluated once: the top-3 ensemble scored 0.989 and the best single seed scored 0.988, both within seed noise, so I kept the simplest option (single model, no TTA). Dropping B10, resnet18 and a higher lr were not explored beyond the screen. About 17 of 40 min were used.

## Test result

- Test accuracy: **0.9830**
- 95% CI: [0.9750, 0.9900] (95% bootstrap CI)
- Scored: 2026-11-01T09:42:10.265813+00:00
