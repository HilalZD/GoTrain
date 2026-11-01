# m-chesapeake v1

- Run: `m-chesapeake-20261031-215710-c678` (best member)
- Model: ensemble of 3 runs with TTA
- Validation miou: **0.7765**
- Extra (not the official score, not used for selection): mIoU excluding the no-data class 0.7291
- Stopped by its time budget before max_epochs (the curve may still rise)
- Training time: 9.9 min, 17 epochs
- Lineage (newest first): `m-chesapeake-20261031-215710-c678`
- Git commit: `b4561313a1b405693452ba0fc309769c173f7b48`
- Registered: 2026-10-31T22:09:51.594155+00:00

## Members

Model: ensemble of 3 runs with TTA. The members' probabilities are averaged and averaged over 8 flip/rot90 views (TTA). The validation score above is the whole model's, measured by evaluate_candidates.

- `m-chesapeake-20261031-215710-c678`: val 0.7678 alone; convnext_tiny/imagenet lr=0.0001 wd=0.01 bs=16 img=256 ep=30 loss=dice_ce aug=flip+rot90 bands=all seed=2
- `m-chesapeake-20261031-214713-3627`: val 0.7670 alone; convnext_tiny/imagenet lr=0.0001 wd=0.01 bs=16 img=256 ep=30 loss=dice_ce aug=flip+rot90 bands=all seed=1
- `m-chesapeake-20261031-204638-6134`: val 0.7663 alone; convnext_tiny/imagenet lr=0.0001 wd=0.01 bs=16 img=256 ep=30 loss=dice_ce aug=flip+rot90 bands=all

## Data

- task_type: segmentation
- sensor: aerial-rgbn
- n_bands: 4
- n_classes: 7
- image_size: [256, 256]
- split_sizes: {'train': 3000, 'val': 1000}

## Config (best member)

```json
{
  "task": "m-chesapeake",
  "backbone": "convnext_tiny",
  "weights": "imagenet",
  "bands": "all",
  "img_size": 256,
  "batch_size": 16,
  "lr": 0.0001,
  "weight_decay": 0.01,
  "max_epochs": 30,
  "loss": "dice_ce",
  "augment": [
    "flip",
    "rot90"
  ],
  "train_subset_frac": 1.0,
  "seed": 2,
  "reason": "Seed check of m-chesapeake-20261031-204638-6134 (seed 2). Original reason: lr contrast: lower lr"
}
```

## Notes

Chosen by mean validation score. Config: U-Net convnext_tiny, imagenet weights, 4 bands, 256px, bs16, lr 1e-4, wd 0.01, dice_ce, flip+rot90, 10 min (~17 epochs). The 3 seeds of this config scored 0.7663/0.7670/0.7678, mean 0.7670 ± 0.0007. The 3-seed ensemble with 8-view TTA scored 0.7765 on val (delta_vs_single_best +0.0087); the same ensemble without TTA scored 0.7750 and a single run with TTA 0.7713. The tool recommended the TTA ensemble, and the 0.0015 TTA gain is larger than the seed std. Other results: lr 5e-5 gave 0.764; lr 3e-4 gave 0.733 in the 4-min screen; ce gave 0.761; resnet34 gave 0.756; color_jitter gave 0.7674, a tie with no jitter (single seed). The best published reference is 0.7081 on test, and our val is above it (val is optimistic; 20 val images duplicate train images). No human baseline. Weakest classes: barren (IoU ~0.56, confused with low vegetation) and impervious roads (~0.60).

## Test result

- Test miou: **0.7573**
- 95% CI: [0.7402, 0.7718] (95% bootstrap CI)
- Scored: 2026-11-01T09:49:59.953333+00:00
