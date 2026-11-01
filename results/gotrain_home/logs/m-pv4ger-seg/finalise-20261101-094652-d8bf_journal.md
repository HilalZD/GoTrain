# m-pv4ger-seg / finalise / finalise-20261101-094652-d8bf

- engine: `real`
- home: `/home/sagemaker-user/gotrain_home`
- model: `eu.anthropic.claude-sonnet-5-5`
- git_commit: `a9781883bb03af72cdad37b6e2b552994fe85ae5`
- max_tool_calls: `12`
- budget: `{'used_min': 95.011, 'remaining_min': 24.989}`

**Agent:** Only v1 is registered, so it's the best by validation score. I'll call final_test on it once. I'm also reading existing lessons in the same step, since the two calls don't depend on each other.

-> `final_test` (14.0 s)
   - args: `{"task": "m-pv4ger-seg", "version": "v1"}`
   - result: `{"task": "m-pv4ger-seg", "version": "v1", "test_score": 0.95459, "metric_name": "miou", "git_commit": "a9781883bb03af72cdad37b6e2b552994fe85ae5", "timestamp": "2026-11-01T09:47:09.105754Z", "ci_low": 0.9466, "ci_high": 0.9613, "ci_level": 0.95, "ci_resamples": 1000, "members": ["m-pv4ger-seg-2026103... [124 chars cut]`

-> `read_lessons` (0.0 s)
   - args: `{"tags": ["segmentation"], "task": "m-pv4ger-seg"}`
   - result: `{"tags": ["segmentation"], "n_found": 10, "lessons": [{"date": "2026-10-31", "task": "m-chesapeake", "tags": ["aerial-rgbn", "diagnose", "segmentation", "source:tune", "workflow"], "lesson": "Harness note: after `diagnose`, every tool call was blocked until the VISUAL OBSERVATION and HYPOTHESIS text... [5111 chars cut]`

-> `leaderboard` (0.0 s)
   - args: `{"task": "m-pv4ger-seg", "top_k": 10}`
   - result: `{"task": "m-pv4ger-seg", "n_runs": 12, "metric_name": "miou", "top": [{"rank": 1, "run_id": "m-pv4ger-seg-20261031-201046-e085", "status": "ok", "val_score": 0.95984, "minutes": 9.94, "config": "convnext_tiny/imagenet lr=0.0002 wd=0.01 bs=16 img=320 ep=30 loss=ce aug=flip+rot90 bands=all seed=2", "p... [3433 chars cut]`

**Agent:** The test result is in. Existing lessons already cover the backbone, loss, budget and seed findings, so I'll write only new ones: the validation-to-test gap and the practical conclusions from it.

-> `write_lesson` (0.0 s)
   - args: `{"task": "m-pv4ger-seg", "lesson": "Final result for m-pv4ger-seg v1 (single U-Net convnext_tiny, ce, lr 2e-4, 320px, no TTA): val mIoU 0.9582 and test mIoU 0.9546 (95% CI 0.9466-0.9613). The gap of 0.0036 is about 2 seed-stds (0.0015) and the val score is inside the test CI, so it is normal selecti... [137 chars cut]`
   - result: `{"task": "m-pv4ger-seg", "saved": true, "tags": ["320px", "aerial-rgb", "m-pv4ger-seg", "segmentation", "source:finalise", "test-gap"], "n_lessons": 23, "budget": {"used_min": 95.011, "remaining_min": 24.989}}`

-> `write_lesson` (0.0 s)
   - args: `{"task": "m-pv4ger-seg", "lesson": "On near-saturated binary aerial segmentation (val mIoU about 0.958), the test CI is about \u00b10.007 wide, which is larger than every tuning gain found (loss, 2x training time, 3-seed ensemble+TTA +0.001). Spend little budget on small tweaks there. The remaining ... [228 chars cut]`
   - result: `{"task": "m-pv4ger-seg", "saved": true, "tags": ["320px", "aerial-rgb", "budget", "m-pv4ger-seg", "segmentation", "source:finalise"], "n_lessons": 24, "budget": {"used_min": 95.011, "remaining_min": 24.989}}`

**Agent:** I tested v1, the only registered version, once. Its validation mIoU was 0.9582 and its test mIoU was 0.9546 (95% CI 0.9466–0.9613).

v1 is a single U-Net with a convnext_tiny backbone (imagenet weights), trained with cross-entropy at 320px, with no test-time augmentation (TTA). Its run is `m-pv4ger-seg-20261031-193048-992b`.

The 0.0036 gap between validation and test is about two seed standard deviations (0.0015), and the validation score sits inside the test confidence interval. I read it as ordinary noise, not overfitting.

I wrote two lessons, both tagged aerial-rgb, segmentation and 320px:
1. The validation-to-test gap above and what it means.
2. On this near-saturated binary segmentation task, tuning gains such as a 3-seed ensemble with TTA (+0.001) are smaller than the test confidence interval (about ±0.007). The remaining errors are dark or low-contrast panels and roof-structure confusions, so context or colour augmentation is a better next step than longer training.

**HARNESS: GoPilot export (after finalise, with test score).** m-pv4ger-seg v1 is now the MCP tool `segment_solar_panels` (bundle `registry/m-pv4ger-seg/v1/gopilot`, hash 7c9b0546045f).

## Session end (2026-11-01 09:47:18 UTC)

- status: `finished`
- budget: `{'used_min': 95.011, 'remaining_min': 24.989}`
- stop_reason: `end_turn`
- continuations: `0`
- registered_by: `None`
- version: `None`
- lessons_written: `0`
- usage: `{'inputTokens': 16411, 'outputTokens': 1381, 'totalTokens': 17792}`

