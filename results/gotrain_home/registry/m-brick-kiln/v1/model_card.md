# m-brick-kiln v1

- Run: `m-brick-kiln-20261031-182603-21e3`
- Model: single run
- Validation accuracy: **0.9897 ± 0.0021** (mean of 3 seeds); single-run score 0.9920
- Training time: 4.9 min, 30 epochs
- Lineage (newest first): `m-brick-kiln-20261031-182603-21e3` <- `m-brick-kiln-20261031-181347-88f9`
- Git commit: `65537a79e384bbdbe0a3496ba25fc7a077827584`
- Registered: 2026-10-31T18:46:30.007695+00:00

## Data

- task_type: classification
- sensor: sentinel-2
- n_bands: 13
- n_classes: 2
- image_size: [64, 64]
- split_sizes: {'train': 15063, 'val': 999}

## Config

```json
{
  "task": "m-brick-kiln",
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
  "train_subset_frac": 0.4,
  "seed": 0,
  "reason": "Halving rung 2 of m-brick-kiln-20261031-181347-88f9 at 6 min. Original reason: upsampling helped on eurosat; kilns are small (10-15px)"
}
```

## Notes

ResNet50, ssl4eo_s2 weights, 13 bands, upsampled to 128px, CE, lr 3e-4, wd 0.01, flip+rot90, batch 64. Chosen by mean validation score: seed group mean 0.9897 ± 0.0021 (3 seeds: 0.992, 0.988, 0.989); the registered original run scored 0.992. Above the best reference (0.9879 test, Prithvi-EO-2.0-600M), though that is a test number and this is validation. No human baseline was given. Ensembles (2 or 3 members) and TTA tied within seed noise (single+TTA 0.993, delta +0.001; 3-ensemble 0.990), so I chose the simpler single model without TTA. Exploration findings: 128px beat 64px (0.991 vs 0.984 in the screen), ssl4eo_s2 beat imagenet (0.991 vs 0.987), and focal tied or lost against CE. Used about 35 of 75 min.

## Test result

- Test accuracy: **0.9820**
- 95% CI: [0.9730, 0.9890] (95% bootstrap CI)
- Scored: 2026-11-01T09:44:43.745903+00:00
