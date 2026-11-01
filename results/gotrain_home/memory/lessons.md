
## 2026-10-31 m-eurosat
tags: classification, img_size, m-eurosat, sentinel-2, source:tune
m-eurosat (2000 train images, 64px, 13 bands): ResNet50 with ssl4eo_s2 weights upsampled to 128px scored 0.986 in a 1.5-min screen and 0.985 over 30 epochs. The 64px ssl4eo_s2 variants scored 0.973-0.979 in the same screen. Upsampling is a cheap way to get more from pretrained filters on small Sentinel-2 chips. It was run once (one change from the 64px baseline), so treat it as indicative.

## 2026-10-31 m-eurosat
tags: classification, m-eurosat, sentinel-2, source:tune, weights
On m-eurosat, ssl4eo_s2 weights clearly beat imagenet weights with the same ResNet50 config at 64px: 0.979 vs 0.946 in the rung-1 screen (1.5 min). Use ssl4eo_s2 on Sentinel-2 tasks, and don't spend budget on imagenet there.

## 2026-10-31 m-eurosat
tags: budget, classification, m-eurosat, sentinel-2, source:tune
m-eurosat training is cheap: ResNet50 at 64px takes about 0.04 min/epoch and at 128px about 0.065 min/epoch. A full 30-epoch run takes about 2 min, so a 6-config halving round with 1.5-min screens finished 6 trials in about 9 min with no epoch warnings. Short screens are enough on small datasets, and the saved budget can go to seed checks.

## 2026-10-31 m-eurosat
tags: classification, ensemble, m-eurosat, seed-noise, sentinel-2, source:tune
Seed noise on m-eurosat validation (1000 images) is about ±0.002: five seeds of the same config scored 0.983-0.988, mean 0.9856. A 3-model ensemble (0.989) and TTA (0.987-0.988) tied with the best single seed (0.988), so they gave no gain beyond noise. Differences under 0.003 between configs should be treated as ties.

## 2026-10-31 m-brick-kiln
tags: classification, img_size, m-brick-kiln, sentinel-2, source:tune
On m-brick-kiln (15k train, 64px, 13 bands, 2 classes), ResNet50 with ssl4eo_s2 weights reached 0.984 in a 2.5-min screen at native 64px, and 0.991 when upsampled to 128px. The 128px run cost 0.175 min/epoch at 40% of the train data, against 0.106 at 64px. Upsampling is worth trying early on small-object Sentinel-2 chips, since kilns are only 10-15 px across.

## 2026-10-31 m-brick-kiln
tags: classification, loss, m-brick-kiln, sentinel-2, source:tune, weights
On m-brick-kiln at 128px with identical config, ssl4eo_s2 weights scored 0.991 and imagenet weights 0.987 in the 2.5-min screen. The gap is small, but ssl4eo_s2 stays the right default for Sentinel-2. Focal loss tied cross-entropy in the screen (0.991 vs 0.991) and was lower after the longer run (0.989 vs 0.992), so stick with ce for this mildly imbalanced (2:1) binary task. Raising lr from 3e-4 to 1e-3 at 64px gave no gain (0.984 vs 0.984).

## 2026-10-31 m-brick-kiln
tags: classification, ensemble, m-brick-kiln, seed-noise, sentinel-2, source:tune
Seed noise on m-brick-kiln validation (999 images) is about ±0.002: three seeds of the same config scored 0.992, 0.988 and 0.989 (mean 0.9897, std 0.0021). The best single seed was the lucky one. Ensembles of 2-3 seeds (0.991, 0.990) and TTA (0.993 single, 0.989 on the 3-model ensemble) all tied within noise. Report the seed mean and skip ensembles and TTA here.

## 2026-10-31 m-brick-kiln
tags: budget, classification, m-brick-kiln, sentinel-2, source:tune
Budget on m-brick-kiln: a 6-config halving round with a 2.5-min screen at train_subset_frac 0.4 and a 6-min rung 2 for the top 2 cost about 24.6 min and ran 14-25 epochs per screen with no warnings. The round already gave 0.992, so a single round plus a 2-seed check (about 10 min) used under half of the 75 min. Cheap screens on a ~15k-image 64px task are enough, and extra rounds are unnecessary when the first one beats the reference.

## 2026-10-31 m-pv4ger-seg
tags: aerial, aerial-rgb, convnext_tiny, m-pv4ger-seg, rgb, segmentation, source:tune
m-pv4ger-seg (aerial RGB, 320px, 3000 train): a U-Net with convnext_tiny and imagenet weights beat resnet18/34/50 in the 4-minute screen (0.9516 vs 0.9468-0.9508). At 10 minutes the two best configs reached 0.9570 (convnext dice_ce) and 0.9562 (resnet34 ce). Try convnext_tiny first on aerial binary segmentation.

## 2026-10-31 m-pv4ger-seg
tags: aerial, aerial-rgb, loss, m-pv4ger-seg, segmentation, source:tune
m-pv4ger-seg: plain ce and dice_ce gave the same score on convnext_tiny (0.9577 vs 0.9570 at 10 minutes, seed std 0.0015). Dice was not needed for this 80/20 imbalance, so loss is a low-priority knob there.

## 2026-10-31 m-pv4ger-seg
tags: aerial, aerial-rgb, budget, m-pv4ger-seg, segmentation, source:tune
m-pv4ger-seg: doubling the convnext_tiny ce run from 10 to 20 minutes (13 to 25 epochs) gave no gain (0.95771 to 0.95768). Validation plateaus at about 0.958 after roughly 13 epochs, so the exploit share is better spent on seed checks than on longer runs.

## 2026-10-31 m-pv4ger-seg
tags: aerial, aerial-rgb, ensemble, m-pv4ger-seg, seeds, segmentation, source:tune
m-pv4ger-seg: three seeds of the best config gave 0.9577, 0.9570 and 0.9598, mean 0.9582 and std 0.0015. A 3-model ensemble with test-time augmentation reached 0.9608 (+0.001), inside the noise, so the single model was registered. The 4-minute screen spread was 0.0048 with 3-6 epochs per trial and was flagged inconclusive, so treat the early ranking as weak.

## 2026-10-31 m-pv4ger-seg
tags: aerial, aerial-rgb, diagnosis, m-pv4ger-seg, rgb, segmentation, source:tune
m-pv4ger-seg error grid: the worst predictions were dark or low-contrast panels that were missed (solar pv recall error 3.4%, versus 0.9% false positives) and fragmented blobby masks. Skylights and other roof structures were confused with panels. A fix was not tested; larger context, a higher-capacity model or colour augmentation are candidate ideas.

## 2026-10-31 m-chesapeake
tags: aerial, aerial-rgbn, convnext_tiny, lr, m-chesapeake, segmentation, source:tune
m-chesapeake (aerial RGBN, 256px, 3000 train, U-Net convnext_tiny imagenet, dice_ce, flip+rot90): lr was the biggest knob in the 4-minute screen. lr 1e-4 scored 0.752, 3e-4 scored 0.733, and focal at 3e-4 scored 0.727. At 10 minutes lr 1e-4 reached 0.7663 and lr 5e-5 reached 0.7640, so the optimum is flat around 5e-5 to 1e-4. Start convnext_tiny segmentation at lr 1e-4 rather than 3e-4.

## 2026-10-31 m-chesapeake
tags: aerial, aerial-rgbn, augmentation, backbone, loss, m-chesapeake, segmentation, source:tune
m-chesapeake backbone and loss comparison at 10 minutes (about 17 epochs for convnext, 22 for resnet34): convnext_tiny dice_ce lr 1e-4 scored 0.7663, convnext_tiny ce 0.7614, and resnet34 dice_ce lr 5e-4 scored 0.7536. resnet34 at lr 2e-4 scored 0.7564. Adding color_jitter to the convnext_tiny lr 1e-4 config scored 0.7674, a tie with the 0.7663 baseline (single seed, so inconclusive). convnext_tiny with dice_ce is a good default, and loss and augmentation tweaks moved the score by less than 0.5 points.

## 2026-10-31 m-chesapeake
tags: aerial-rgbn, ensemble, m-chesapeake, seeds, segmentation, source:tune, tta
m-chesapeake seed check: three seeds of the best config scored 0.7663, 0.7670 and 0.7678, a mean of 0.7670 and std 0.0007, so seed noise is very small. A 3-seed ensemble scored 0.7750 (+0.0072 over the best single run) and the ensemble with 8-view TTA scored 0.7765 (+0.0087). A single run with TTA scored 0.7713. Because noise is that small, a 3-seed ensemble with TTA gives a real gain of about 1 point, so budget for it when seed std is small.

## 2026-10-31 m-chesapeake
tags: aerial, aerial-rgbn, diagnosis, m-chesapeake, segmentation, source:tune
m-chesapeake diagnosis: weakest classes are barren (IoU 0.56, 22% of its pixels predicted as low vegetation) and impervious roads (0.60, confused with impervious other). The worst tiles are large quarry or barren areas labelled as vegetation, and masks sometimes have hand-drawn straight edges that do not follow the imagery. No fix for barren was tested, so these errors are partly label ambiguity. Ideas not yet tried are higher resolution and class-weighted loss.

## 2026-10-31 m-chesapeake
tags: aerial-rgbn, diagnose, segmentation, source:tune, workflow
Harness note: after `diagnose`, every tool call was blocked until the VISUAL OBSERVATION and HYPOTHESIS text was accepted. Five calls were blocked before one went through. The call that succeeded had the observation and hypothesis written into the config `reason` fields, starting with 'Visual observation: ... Hypothesis: ...'. Do this from the first call after diagnose to avoid wasting tool calls.

## 2026-11-01 m-eurosat
tags: 128px, 64px, classification, m-eurosat, sentinel-2, source:finalise
Final test for m-eurosat (ResNet50, ssl4eo_s2, 128px, 30 ep, ce, flip+rot90, all bands, single model, no TTA) scored 0.983 accuracy (95% CI 0.975-0.990) against a val score of 0.9856 (5-seed mean, std 0.0019). The 0.003 gap is inside the CI and inside seed noise. Quote the seed-mean val and the test CI rather than the best single seed (0.988).

## 2026-11-01 m-eurosat
tags: 64px, classification, m-eurosat, sentinel-2, source:finalise
On m-eurosat at 64px, backbone and band choices barely mattered in 30-epoch screens. ResNet18 scored 0.978 (0.85 min), ResNet50 0.978 (1.33 min) and ResNet50 on 12 bands 0.979. Raising lr to 1e-3 dropped ResNet50 to 0.973. The 128px upsampling (0.985-0.988) was the only change that beat noise, so spend tuning budget on input size before backbone or bands. ResNet18 is a cheap fallback.

## 2026-11-01 m-brick-kiln
tags: 128px, 64px, classification, m-brick-kiln, sentinel-2, source:finalise
Final test for m-brick-kiln (ResNet50 ssl4eo_s2, 128px upsampled from 64px, 30 ep, ce, flip+rot90, all 13 bands, single model, no TTA) scored 0.982 accuracy (95% CI 0.973-0.989). Registered val was 0.9897 (3-seed mean; best seed 0.992). The 0.008 gap to the seed-mean val is slightly larger than seed noise (±0.002) but the val score sits at the edge of the test CI, which is consistent with mild selection optimism on val.

## 2026-11-01 m-brick-kiln
tags: 64px, classification, m-brick-kiln, seed-noise, sentinel-2, source:finalise
Expect test accuracy about 0.5-1 point below the best val score on ~1k-image binary Sentinel-2 splits. The registered run was the lucky seed (val 0.992 against seed mean 0.9897), and test came in at 0.982. Plan on the seed-mean val minus about 0.005-0.01, and don't chase val differences under 0.003.

## 2026-11-01 m-pv4ger-seg
tags: 320px, aerial-rgb, m-pv4ger-seg, segmentation, source:finalise, test-gap
Final result for m-pv4ger-seg v1 (single U-Net convnext_tiny, ce, lr 2e-4, 320px, no TTA): val mIoU 0.9582 and test mIoU 0.9546 (95% CI 0.9466-0.9613). The gap of 0.0036 is about 2 seed-stds (0.0015) and the val score is inside the test CI, so it is normal selection and sampling noise, not overfitting.

## 2026-11-01 m-pv4ger-seg
tags: 320px, aerial-rgb, budget, m-pv4ger-seg, segmentation, source:finalise
On near-saturated binary aerial segmentation (val mIoU about 0.958), the test CI is about ±0.007 wide, which is larger than every tuning gain found (loss, 2x training time, 3-seed ensemble+TTA +0.001). Spend little budget on small tweaks there. The remaining errors are dark or low-contrast panels and roof-structure confusions, so try context or colour augmentation rather than more epochs.

## 2026-11-01 m-chesapeake
tags: 256px, aerial-rgbn, m-chesapeake, segmentation, source:finalise, test-gap
Final m-chesapeake v1 (3-seed U-Net convnext_tiny dice_ce ensemble + 8-view TTA, 256px): val mIoU 0.7765, test mIoU 0.7573 (95% CI 0.7402-0.7718). The gap of 0.019 is about 27 seed-stds (0.0007), so the seed std badly understates the real uncertainty. The val score sits just above the test CI upper bound. Expect about 2 points of val-to-test shrinkage on this multi-class task, and judge gains smaller than about 0.015 against the CI width, not the seed std.

## 2026-11-01 m-chesapeake
tags: 256px, aerial-rgbn, budget, ensemble, m-chesapeake, segmentation, source:finalise
m-chesapeake: all registered configs were stopped_by_budget at about 10 min (about 17 epochs for convnext_tiny), so validation may still have been rising. Within that limit, ensemble+TTA gave +0.0087 on val. Test CI half-width was about 0.016, so most of this gain is not separable from noise on test. For similar multi-class aerial tasks, test untried levers with bigger expected effect (higher resolution, class-weighted loss for barren and roads, longer training) before further seed or TTA stacking.
