# m-brick-kiln v1

- Run: `m-brick-kiln-20261031-222158-a214`
- Model: single run
- Validation accuracy: **0.9900 ± 0.0017** (mean of 3 seeds); single-run score 0.9910
- Stopped by its time budget before max_epochs (the curve may still rise)
- Training time: 4.9 min, 23 epochs
- Lineage (newest first): `m-brick-kiln-20261031-222158-a214` <- `m-brick-kiln-20261031-221212-f0eb`
- Git commit: `b4561313a1b405693452ba0fc309769c173f7b48`
- Registered: 2026-10-31T22:42:12.454098+00:00

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
  "img_size": 64,
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
  "reason": "Halving rung 2 of m-brick-kiln-20261031-221212-f0eb at 5 min. Original reason: bigger SSL4EO backbone"
}
```

## Notes

ResNet50 with ssl4eo_s2 weights, all bands, 64px, CE, lr 3e-4, wd 0.01, flip+rot90, 5 min budget. Chosen by mean validation score across 3 seeds: 0.990 ± 0.0017 (single run 0.991). Ensembles of 2-3 seeds and TTA tied within the seed noise (delta 0 to -0.001), so the simplest option (single model, no TTA) was registered. Validation 0.990 is at or above the published test reference (0.9879, Prithvi-EO-2.0-600M); validation is usually slightly optimistic. No human baseline was given. Explored: resnet18 vs resnet50, 128px upsampling, imagenet weights, higher lr, focal loss. Spread in the 2 min screen was small (0.985-0.990).
