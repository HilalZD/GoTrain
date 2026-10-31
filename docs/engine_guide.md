# Engine guide: Serena's files, and how to defend them

This guide covers the ML and data side of GoTrain: what each file does, why it's built that way,
and short answers to questions judges are likely to ask. Details and numbers are in
`docs/data_notes.md`. Contract questions with Hilal are in `docs/contract-requests.md`.

**The big picture in one paragraph.** The agent never writes training code. It picks a *config*
(backbone, weights, bands, learning rate, loss, augmentations…). A fixed, tested engine runs it
on the training data and scores it on the validation data. The engine is four modules:
- `data.py` reads the data;
- `models.py` builds the network;
- `train.py` trains and scores;
- `diagnostics.py` explains the errors.

The test split is opened in exactly one place, and only when the final-test tool asks.

---

## `gotrain/engine/data.py`: reading GEO-Bench safely

**What it does.** It finds a task on disk, opens the train or val split, and turns every sample
into an array with correct band and class names. It also measures each task once (`inspect_task`)
and gives the agent a summary card of the dataset (`describe_dataset`).

**Key decisions and why**
- **Errata tables instead of special cases.** The published data has wrong band names (eurosat,
  brick-kiln) and wrong or missing class names (eurosat, chesapeake). The fixes live in small
  tables (`BAND_ORDER_FIXES`, `CLASS_NAME_FIXES`), each with a comment citing our evidence. The
  rest of the code is task-agnostic.
- **The test split is locked in code.** `open_split()` refuses `"test"` unless
  `allow_test=True`, and only `evaluate()` passes that, which only the final-test tool calls.
  A test fails the build if any other file asks for the test split.
- **Normalisation statistics come from TRAIN only.** Mean and std are learned numbers, like
  weights. Computing them on val or test would leak information about held-out images.
- **The card keeps facts that simple fields would lose.** Examples: "B10 is nearly empty",
  "always predicting 'not kiln' already gets 66%", the ground resolution. These go in its notes,
  so the agent sees them.

**Likely questions**
- *Why not use GEO-Bench's own `band_stats.json` for normalisation?* We can't verify which
  splits it was computed on. Our stats are provably train-only.
- *How do you know the test split is never touched?* It's guarded in code. `open_split` raises
  without `allow_test`. A static test checks that only `train.py` passes it, and a second test
  checks that only `final_test.py` requests it.
- *What happens with a task you've never seen?* Nothing in the code is task-specific except the
  errata tables, so the card, the dataset and training work unchanged ("zero-touch").

## `gotrain/engine/models.py`: the model menu

**What it does.** It builds a classifier (timm) or a U-Net (segmentation_models_pytorch) for any
number of input bands, with ImageNet weights, Sentinel-2 SSL4EO weights, or random weights. Which
backbone and weights combinations are allowed comes from Hilal's `combos.py`, the single list
shared with the agent.

**Key decisions and why**
- **First-layer adaptation.** ImageNet models expect 3 channels; our data has 3, 4 or 13. The
  libraries repeat the RGB filters across all channels and rescale them, so pretrained features
  still work.
- **Satellite-pretrained weights.** TorchGeo's SSL4EO MoCo ResNets were pretrained on 13 Sentinel-2
  bands, divided by 10000. We feed them exactly that: bands reordered *by name*, missing bands
  (brick-kiln: B06, B09, B10) zero-filled, input / 10000. We checked the band order in TorchGeo's
  installed source code, not from memory.
- **Each checkpoint stores its own normalisation.** That's how ensembles can mix ImageNet and
  SSL4EO models.

**Likely questions**
- *Why zero-fill missing bands instead of dropping filters?* The model always sees the 13 channels
  it was pretrained on. A zero channel contributes nothing, so it's safe and easy to audit.
- *Why not z-score the SSL4EO input like the others?* Those weights learned on reflectance / 10000.
  Changing the input scale would make the pretrained features meaningless.

## `gotrain/engine/train.py`: training and scoring

**What it does.** `train()` runs one config inside a time budget and scores every epoch on the full
validation set. It keeps the best checkpoint and never crashes the caller: problems come back as
`status="error"` with a hint. `evaluate()` scores one model or an ensemble on val (or on test, for
the final test), and `predict()` labels new images.

**Key decisions and why**
- **The learning-rate schedule follows the clock.** The cosine schedule's progress is the larger of
  "time used / time available" and "steps done / max steps". A 2-minute scouting trial and a
  30-minute final run both finish their schedule properly.
- **Time is reserved for the last validation.** It's estimated from real batch speed, ignoring the
  slow warm-up batch, and capped at half the budget. This fixed a real bug found on SageMaker: a
  slow first batch made the engine stop after 10 seconds.
- **The same validation set every time** (never subsampled), so scores are comparable across trials.
- **One probability function for everything.** Training validation, `evaluate()` and `predict()` all
  use `class_probs()`, so `evaluate(val)` returns exactly the score `train()` reported (tested).
- **Ensembles and TTA average probabilities.** TTA means averaging over the 8 flips and rotations;
  satellite images have no "up".
- **Metrics follow GEO-Bench:** accuracy for classification, mIoU for segmentation, computed at
  dataset level (pixel counts summed over all images per class, then averaged over classes).

**Likely questions**
- *What if a trial runs out of time?* It stops cleanly, keeps its best checkpoint and returns
  `"ok"` if at least one full epoch finished. If not even one epoch fits, it returns a "0-epoch
  timeout", which the tools treat as "budget too small".
- *How do you pick TTA or an ensemble without peeking at test?* Compare `evaluate(..., "val")`
  with and without them, and pass the same choice to the final test.
- *Why mIoU and not pixel accuracy for solar panels?* 80% of pixels are background, so predicting
  "no panel" everywhere already gets 80% accuracy. mIoU punishes that.

## `gotrain/engine/diagnostics.py`: explaining the errors

**What it does.** `diagnose()` reads a finished run's saved validation predictions and returns:
per-class scores, a confusion matrix, a verdict on the learning curve (overfitting, underfitting
or fine), and a picture of the worst predictions (`errors.png`).

**Key decisions and why**
- **The agent looks at its mistakes, like an engineer.** Claude can see images, so the grid
  (image | truth | prediction) lets it reason, e.g. "it paints whole roofs, so keep full
  resolution".
- **No re-training or re-scoring.** It reuses the predictions saved at the best epoch, so it's
  fast and matches the reported score.
- **Same output keys as the fake engine**, so the agent side didn't need changes.

**Likely questions**
- *How is "overfitting" decided?* Val peaked in the first 75% of epochs and then fell by more than
  0.01. "Underfitting" means it was still rising at the end. It's the same rule as the fake engine.
- *What's `miou_excl_nodata`?* For chesapeake, an extra number: mIoU without the blank "no data"
  class. The official score keeps all 7 classes, to stay comparable with GEO-Bench.

## `scripts/baseline_eurosat.py` and `scripts/baseline_pv4ger_seg.py`: hand baselines

**What they do.** Plain, readable training scripts that set a "human expert" reference before the
agent runs: SSL4EO ResNet50 on eurosat (0.986 val accuracy), and a U-Net with an ImageNet ResNet-34
on pv4ger-seg (0.959 val mIoU). `--smoke` runs a 1-minute check on a laptop CPU.

**Key decisions and why**
- **They show what the agent should reach.** Both beat GEO-Bench's best published val scores
  (0.982 and 0.947); see the table in `data_notes.md`.
- **Upsizing 64 → 224 px for eurosat**, because the pretrained weights saw 224 px crops.
- **Full resolution for solar panels, Dice + BCE loss**, because thin panel edges vanish when
  downscaled and Dice ignores the 80% background.

**Likely question**
- *Did the agent just copy your baseline?* No. The agent only sees configs, scores and lessons. The
  baselines are a reference to judge the agent against.

## `scratch/explore.py`: data reconnaissance

**What it does.** For each task it prints a readable summary and writes the stats file and card. It
also draws sample images, plus one row of tiles per class to check the class names by eye.

**Key decision.** It only prints and plots. All measuring lives in `data.py`, so the numbers the
agent sees and the numbers we checked are the same code.

**Likely question**
- *How did you check the class names?* With pictures and physics: 6 tiles per class plus the mean
  NDVI (a greenness index). Water has negative NDVI and forest the highest. That's how we caught
  eurosat's mislabelled classes.

## `scratch/download_tasks.py`: downloading only what we need

**What it does.** It downloads single task zips from GEO-Bench's Hugging Face repo instead of the
whole 17.5 GB benchmark, and unzips them where geobench expects them.

**Key decisions.** It checks the task name against the live file list. It blocks malicious zip
paths (a zip-slip guard, which we fixed to work on Windows too) and skips tasks already present.

**Likely question**
- *Why not geobench's own downloader?* It has no per-task option and downloads everything
  (17.5 GB zipped, ~65 GB during unzip).

---

## Data bugs we found (and the evidence)

1. **eurosat and brick-kiln band names are wrong in the published data.**
   - *Evidence:* the GEO-Bench README "Known issues" (issues #28, #29).
   - *Fix:* `BAND_ORDER_FIXES` gives the true order. In eurosat, B8A is the *last* channel.
   - *Why it matters:* without the fix, choosing bands by name silently picks the wrong channel.

2. **eurosat class names don't match the labels.**
   - *What's wrong:* the label integers follow the original alphabetical EuroSAT order, but
     geobench lists the names in a different order.
   - *Evidence:* `docs/figures/m-eurosat_per_label.png` shows 6 tiles for each label, and every
     row matches the alphabetical name. Mean NDVI per label confirms it: label 9 is −0.25 (only
     water is negative), label 1 is +0.71 (forest).
   - *Fix:* `CLASS_NAME_FIXES`. Accuracy was never affected; per-class reports and lessons were.

3. **chesapeake class 0 is "no data", and the class names were missing.**
   - *Evidence:*
     - GEO-Bench's converter reads the masks through TorchGeo with `class_set=7`, which keeps the
       raw Chesapeake values (1 water … 6 roads).
     - TorchGeo fills areas outside the imagery with 0.
     - Class 0 is exactly 125 whole blank tiles (8,192,000 px = 125 × 256 × 256), and the imagery
       there is all zeros.
     - In the sample figure, the water tile is class 1.
   - *Fix:* names via `CLASS_NAME_FIXES`. GEO-Bench's own list is off by one: it puts "no-data"
     last. Decision: keep the official 7-class score, and report `miou_excl_nodata` as an extra.

4. **brick-kiln has a shortcut, and accuracy flatters models.**
   - *Evidence:*
     - 66.4% of tiles are "not kiln", so always guessing that already scores 66.4%.
     - Almost-flat tiles (water, cloud, haze) are 10.8% of "not kiln" tiles and 0% of "kiln"
       tiles, so "flat ⇒ not kiln" is a free win.
     - Mean NDVI is the same for both classes (0.40 vs 0.38): kilns are small objects, not a
       land-cover type.
   - *Fix:* the card tells the agent all this, and to judge models by kiln-class F1/recall.

Also checked and noted: chesapeake has 20 val images that are byte-identical to train images (val
is slightly optimistic), and pv4ger-seg has almost no panel-free tiles (panels are 20% of pixels).
