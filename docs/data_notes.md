# GoTrain data notes

Owner: Serena (ML & data). Branch: `serena/data`. Files: `scratch/`, `stats/`, `docs/data_notes.md`, `docs/figures/`.

Status: scripts tested on fake data (see "How this was tested"), then **run on the real m-eurosat and m-pv4ger-seg data on 2026-10-31** (local Windows, Python 3.12.10, geobench 1.1.0, numpy 2.4.6, torch 2.13.0+cpu untouched). Section 4 holds the measured numbers.

---

## 1. Quick start on the SageMaker instance

```bash
# 0. Check what we already have. Do this BEFORE installing anything.
python -c "import sys, numpy; print(sys.version); print('numpy', numpy.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 1. Dry run first: confirm pip will NOT touch torch, and see which geobench version it picks.
pip install --dry-run geobench 2>&1 | tail -5
pip install --dry-run geobench 2>&1 | grep -i torch   # expect no output
# If torch appears in that list: STOP and tell the team.

# 2. Install (base package only; never "geobench[torch]", which pulls torch)
pip install geobench matplotlib

# 3. Download ONLY our tasks (~580 MB of zips total)
python scratch/download_tasks.py m-eurosat m-pv4ger-seg --data-dir data
export GEO_BENCH_DIR="$(pwd)/data"     # see gotcha 3 about notebooks

# 4. Recon: prints summaries, writes stats/ and docs/figures/
python scratch/explore.py m-eurosat m-pv4ger-seg --data-dir data
```

**Python version warning.** `geobench` 1.1.0 (latest, Oct 2026) requires **Python 3.12+**. If the instance runs 3.10 or 3.11, pip will either fail or install an older geobench release. If that happens, run `pip show geobench` and tell the team which version was installed. `explore.py` only uses long-standing functions (`load_task_specs`, `get_dataset`, `pack_to_3d`), but the band-order warning and the safe unpickler only exist in recent versions, so the gotchas below matter even more there.

---

## 2. How the single-task download works (verified, not guessed)

- `geobench`'s own `geobench-download` command (`geobench/geobench_download.py`) has **no per-task option**. It calls `HfApi().list_repo_files("recursix/geo-bench-1.0")` and downloads **every** file. That's ~17.5 GB zipped and ~65 GB of free space during unzip. **Never run it.**
- The Hugging Face dataset `recursix/geo-bench-1.0` stores **one zip per task**:
  - `classification_v1.0/`: `m-bigearthnet.zip` 6.56 GB, `m-brick-kiln.zip` 1.58 GB, **`m-eurosat.zip` 276 MB**, `m-forestnet.zip` 1.92 GB, `m-pv4ger.zip` 1.03 GB, `m-so2sat.zip` 964 MB, `normalizer.json`
  - `segmentation_v1.0/`: `m-NeonTree.zip` 601 MB, `m-SA-crop-type.zip` 1.43 GB, `m-cashew-plant.zip` 1.33 GB, `m-chesapeake.zip` 1.25 GB, `m-nz-cattle.zip` 327 MB, **`m-pv4ger-seg.zip` 300 MB**, `normalizer.json`
  - `normalizer.json` is only used by geobench's plotting tools to rescale published scores. We don't need it.
- `scratch/download_tasks.py` does what the official script does, but for one file:
  1. Looks up `<benchmark>/<task>.zip` in the live repo listing, so the task name is verified.
  2. Calls `hf_hub_download(repo_id="recursix/geo-bench-1.0", filename=..., repo_type="dataset", local_dir=data)`.
  3. Unzips it next to the zip, with a zip-slip guard, then deletes the zip.
  4. Checks that `data/<benchmark>/<task>/task_specs.pkl` exists.
- Result layout (what geobench expects under `$GEO_BENCH_DIR`):
  ```
  data/classification_v1.0/m-eurosat/    task_specs.pkl, default_partition.json, band_stats.json, *.hdf5
  data/segmentation_v1.0/m-pv4ger-seg/   ...
  ```
- Re-running is safe: a task that already has `task_specs.pkl` is skipped.
- Hugging Face flags the zips as containing pickles. geobench reads them with its own restricted unpickler (`geobench._safe_pickle.safe_load`). Always load task specs through `geobench.load_task_specs()`, **never** with plain `pickle.load`.

---

## 3. Outputs and formats

### `stats/<task>.json`: normalisation statistics (read by `train.py`)
```json
{"task": "m-eurosat", "split": "train", "n_samples": 2000, "seed": 0,
 "bands": ["B01", "B02", "..."], "mean": [ ... ], "std": [ ... ], "excluded_pixels": 0}
```
- `bands`, `mean` and `std` are **parallel lists in channel order**: `mean[i]` and `std[i]` belong to `bands[i]`, which is channel `i` of the array.
- Computed on the **train split only**, over at most 2,000 randomly chosen train images (seed 0). It uses a streaming algorithm: per-image mean and variance are merged with Chan's parallel formula and accumulated in float64.
- `std` is the population standard deviation of pixel values.
- Excluded pixels: NaN/inf values, and "nodata" pixels where **every** band is exactly 0. `excluded_pixels` (a single int, matching `BandStats` in `gotrain/contracts.py`) is the total number of sampled train pixels that were nodata or had at least one NaN/inf value. A NaN in one band only removes that band's value; the pixel still counts toward the other bands.
- Usage in training: `x = (x.astype(float32) - mean[:, None, None]) / std[:, None, None]` after transposing to C×H×W.

### `stats/<task>_card.json`: full DatasetCard (future `inspect_dataset` tool)
Shared contract keys: `task, task_type, sensor, bands, n_classes, class_names, image_size, split_sizes (train/val), class_distribution, band_mean, band_std, notes`.
Recon extras: `dtype, value_range (min/max/p1/p99 per band), label_format, mask_unique_values, nodata_fraction, constant_bands, stats_n_samples, spatial_resolution_m, native_band_resolution_m, band_names_corrected, class_names_corrected, excluded_pixels, example_sample`.

**Known mismatch with `DatasetCard` (`gotrain/contracts.py`):** the shared keys line up, but four recon extras have richer shapes than the contract allows, so `DatasetCard.model_validate(card)` currently fails. `class_distribution` is `{name: {count, pct}}` (the contract wants `{name: pct}`), `dtype` is a list, `value_range` is per band (the contract wants one `(min, max)`), and `mask_unique_values` is a dict (the contract wants a list of ints). `describe_dataset()` in `gotrain/engine/data.py` flattens them (M1): `class_distribution` → fractions, `dtype` → one string, `value_range` → (min over bands, max over bands), `mask_unique_values` → sorted list. Facts the flat fields would lose go into `notes` (corrected band/class names, ground resolution, nearly empty bands such as B10, the unit of `class_distribution`). `stats/<task>.json` validates against `BandStats`.

**Where the code lives (since M1):** all measuring and loading logic, including `inspect_task` and the errata tables `BAND_ORDER_FIXES` / `CLASS_NAME_FIXES`, is in `gotrain/engine/data.py`. `scratch/explore.py` only prints and plots on top of it. Sensor names in cards are slugs (`sentinel-2`, `aerial-rgb`) because `tools/inspect.py` and memory lessons match on them.

### `docs/figures/<task>_per_label.png`
Single-label classification only: one row per label index, 6 random train tiles each (seed 0). Each row is titled with the class name and the mean NDVI over up to 200 train tiles of that label (only when red and NIR bands are found by name). When a class-name errata entry is applied, the row also shows geobench's original name. This figure is the evidence for `CLASS_NAME_FIXES`.

`inspect_task(task_name)` in `scratch/explore.py` returns this same dict, so it can become the agent's tool as-is.

### `docs/figures/<task>_samples.png`
8 random train samples (seed 0). RGB bands are picked **by name**, with a 2–98 percentile stretch. Classification tiles are titled with the class name. Segmentation shows each image next to its mask: binary masks as a red overlay titled with the % of positive pixels, multi-class masks drawn directly with a legend, and ignore values in grey.

---

## 4. Per-task results

### m-eurosat (classification)
Known from the brief and the source:
- 10 land-cover classes; 2,000 / 1,000 / 1,000 train / val / test.
- Sentinel-2, 13 bands, 10 m resolution.
- Label is a single integer class index.
- **Band names in the published data are wrong** (see gotcha 2); `explore.py` applies the corrected order.

Measured (`explore.py`, seed 0, stats on all 2,000 train images):
- **Image size:** 64×64 px, 13 bands, all stored at the same size (no resampling needed). 10 m/pixel, so one tile covers 640 m × 640 m.
- **Dtype:** int16 (reflectance × 10,000).
- **Value ranges (train, p1–p99):** most bands sit between ~100 and ~5,400. Train means range from 751 (B09) to 2,665 (B8A); the NIR bands (B08, B8A) and red edge (B06, B07) are the brightest and most variable (std 850–1,220). **B10 (cirrus) is near zero over land:** mean 12.3, std 4.8, p1–p99 5–29, max 90. That band is why the script reports a ~185× scale gap (p99: B10 29 vs B8A 5352).

  | band | B01 | B02 | B03 | B04 | B05 | B06 | B07 | B08 | B09 | B10 | B11 | B12 | B8A |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | train mean | 1360 | 1126 | 1055 | 957 | 1220 | 2052 | 2434 | 2360 | 751 | 12.3 | 1849 | 1131 | 2665 |
  | train std | 251 | 340 | 397 | 593 | 555 | 853 | 1082 | 1115 | 405 | 4.8 | 979 | 745 | 1224 |
  | p1 | 977 | 700 | 461 | 243 | 204 | 182 | 162 | 133 | 58 | 5 | 23 | 12 | 121 |
  | p99 | 2096 | 2260 | 2352 | 2845 | 2830 | 3923 | 4928 | 4858 | 1884 | 29 | 4145 | 3070 | 5352 |
- **Class distribution:** perfectly balanced, 200 train samples per class (10.00% each). Plain cross-entropy and accuracy are fine.
- **Class names corrected** (gotcha 14). geobench's `class_names` list is out of order; `explore.py` now uses the verified label order `0 Annual Crop, 1 Forest, 2 Herbaceous Vegetation, 3 Highway, 4 Industrial Buildings, 5 Pasture, 6 Permanent Crop, 7 Residential Buildings, 8 River, 9 Sea and Lake`.
- **Mean NDVI per label (train, 200 tiles each):** Annual Crop +0.39, Forest +0.71, Herbaceous Vegetation +0.38, Highway +0.45, Industrial +0.22, Pasture +0.65, Permanent Crop +0.36, Residential +0.33, River +0.34, Sea and Lake −0.25.
- **Nodata:** 0.0000% (no all-zero pixels, no NaN/inf). **Constant bands:** none.
- **Duplicates / leakage:** none. 0 exact duplicates among the 2,000 train images, and 0 of the 1,000 val images match a train image byte for byte.
- **Notes from the script:** band names corrected (errata); class names corrected (errata); ~185× band-scale gap (B10 vs B8A), so per-band normalisation is mandatory.
- **Excluded pixels:** 0 (`excluded_pixels` in `stats/m-eurosat.json`).
- **Sample figure:** the tiles are real Sentinel-2 landscapes (fields, rivers, housing grids, industrial roofs), and with the class-name fix the titles now match the content. `m-eurosat_per_label.png` shows 6 tiles for every label. Uniform tiles (water, dense forest) look like coloured static under the 2–98% stretch: the stretch amplifies tiny sensor noise when a tile has almost no contrast. That's a display artefact, not bad data.

### m-pv4ger-seg (segmentation)
Known from the brief:
- Binary segmentation, solar panel vs. background.
- 3,000 / 403 / 403 train / val / test.
- Aerial RGB at 0.1 m resolution.

Measured (`explore.py`, seed 0, pixel stats on 2,000 of 3,000 train images, labels on all 3,000):
- **Image size:** 320×320 px, 3 bands (Red, Green, Blue), 0.1 m/pixel, so one tile covers 32 m × 32 m (a roof or two).
- **Dtype:** uint8, 0–255. Cast to float32 before normalising.
- **Value ranges (train):** Red mean 131.1 / std 54.8, Green 137.3 / 51.3, Blue 139.4 / 48.9; p1–p99 roughly 29–242 in every band. Ordinary photo statistics, close to ImageNet's (×255), so ImageNet weights are a good start.
- **Mask:** 320×320 uint8, values **{0, 1}** (`is_0_1=True`, not 0/255), no ignore values. Use it as-is.
- **Panel pixel %:** **19.96%** of train pixels are `solar pv` (61,316,545 of 307,200,000); 80.04% are `no solar pv`. Per tile: only **6 of 3,000 tiles (0.2%) have no panel at all**; per-tile panel coverage quantiles are 10% → 2.4%, 25% → 6.0%, median 15.4%, 75% → 27.8%, 90% → 45.6%, max 100%. (Per-tile numbers come from a one-off check reading train labels only; it isn't part of `explore.py`.)
- **Nodata:** 0.0000%. **Constant bands:** none.
- **Duplicates / leakage:** none. 0 exact duplicates in the 2,000 sampled train images, and 0 of the 403 val images match one. The check covers only 2/3 of train (the 2,000-image sample).
- **Notes from the script:** none.
- **Sample figure:** the masks sit tightly on the panels in all 8 tiles, including thin panel strips and panels cut off at the tile edge. They follow the panel arrays, not the whole roof.
- Sample names are `lon,lat` coordinates (e.g. `7.001829544,52.21933747`, in north-west Germany). That means tiles could be grouped spatially later if we want to check geographic leakage.

---

## 5. Gotchas for training (things `train.py` must handle)

1. **`get_dataset()` silently gives you 3 bands.** `TaskSpecifications.get_dataset(band_names=...)` defaults to `("red", "green", "blue")`, so a 13-band dataset quietly becomes RGB-only. On a dataset with no RGB alt names it crashes instead. Always pass `band_names=[bi.name for bi in task.bands_info]` (as `_open_split` does).
2. **m-eurosat and m-brick-kiln have mislabelled channels** (geo-bench README "Known issues", issues #28/#29). The pixel data is fine, but the stored names are wrong for most channels, so **selecting by stored name returns the wrong channel**. True order:
   - m-eurosat: `B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A` (note **B8A is last**)
   - m-brick-kiln: `B01 B02 B03 B04 B05 B07 B8A B08 B11 B12 TCI_R TCI_G TCI_B`. B06, B09 and B10 are absent. The last 3 are **8-bit true-colour channels (0–255)** on a totally different scale from the reflectance bands.
   - Channels 0–4 (B01–B05, including RGB) are labelled correctly in both, so RGB composites look right. Any band selection beyond those must use the corrected list. `stats/<task>.json` already stores the corrected names.
3. **`$GEO_BENCH_DIR` is read once, at `import geobench`.** Setting `os.environ[...]` afterwards does nothing. In Jupyter, set it before the kernel starts, or avoid relying on it: `explore.py` locates the task folder itself and calls `geobench.load_task_specs(path)`. Do the same in `train.py`.
4. **Split names are `"train"`, `"valid"` and `"test"`** (not "val"). `get_dataset(split=None)` returns **all splits including test**, so never call it without a split. `explore.py` refuses anything except train/valid. `train.py` should guard the same way, so only the locked `final_test` tool can open test.
5. **Arrays are channels-last.** `sample.pack_to_3d()` returns **H×W×C**; PyTorch wants **C×H×W**, so transpose.
6. **Integer dtypes.** Sentinel-2 bands are int16; aerial RGB is likely uint8. Cast to float32 before normalising, because uint8 arithmetic overflows silently.
7. **Different native resolutions.** Sentinel-2 bands are captured at 10, 20 and 60 m. If a dataset stores them at different sizes, `pack_to_3d(resample=True)` upsamples them (cubic) to the largest. The card notes when this happens; training must use the same resampling.
8. **Pixel imbalance in pv4ger-seg is moderate, not severe (measured).** Panels are 19.96% of train pixels (a 4:1 ratio), because tiles are cropped around known installations. Plain BCE will train fine, but Dice + BCE still helps on the many tiles with only a few % panel. Report IoU of the **panel** class, not pixel accuracy (predicting "no panel" everywhere already scores 80%).
9. **Mask encoding.** Check `mask_unique_values`. If masks are 0/255, map 255→1. Any value ≥ `n_classes` is an "ignore" value and must become the loss's `ignore_index`.
10. **Nodata rule.** Pixels where every band is exactly 0 are treated as nodata and excluded from the stats. Training should do the same, or at least know they exist (the card reports the fraction).
11. **Don't normalise with geobench's own `band_stats.json`.** Each task ships one, but we can't verify which splits it was computed on. We use our own train-only stats.
12. **Duplicate and leakage check is sample-based.** It compares exact byte hashes of the ≤2,000 sampled train images against ≤2,000 val images. Near-duplicates (shifted or recompressed images) are not detected.
13. **Upstream multilabel quirk.** `MultiLabelClassification` stores names in `class_name` (typo), so `.class_names` is empty. `explore.py` handles it; reuse `_class_names()` for m-bigearthnet.
14. **m-eurosat class names are out of order upstream; corrected via `CLASS_NAME_FIXES` (found and verified 2026-10-31).** geobench lists `Industrial, Residential, Annual Crop, Permanent Crop, River, Sea and Lake, Herbaceous Vegetation, Highway, Pasture, Forest`, but the label integers follow the **original alphabetical EuroSAT order**: `0 Annual Crop, 1 Forest, 2 Herbaceous Vegetation, 3 Highway, 4 Industrial Buildings, 5 Pasture, 6 Permanent Crop, 7 Residential Buildings, 8 River, 9 Sea and Lake`. Evidence, train split only:
    - **`docs/figures/m-eurosat_per_label.png`** shows 6 random train tiles for every label index. Before the fix, all 10 rows matched the alphabetical name, not the geobench name: 0 large ploughed/striped fields, 1 dense canopy, 2 scrub and sparse vegetation on rocky slopes, 3 motorways and an interchange, 4 large roofs and a stadium, 5 uniform grassland, 6 small strip parcels (orchard/vineyard style; the least distinctive row, but consistent and the only label left), 7 dense housing, 8 a river in every tile, 9 flat water.
    - **Mean NDVI per label** (≤200 tiles each): label 9 is −0.25 (only water gives negative NDVI), label 1 is +0.71 (the highest, i.e. forest), label 5 is +0.65 (grassland), and label 4 is +0.22 (the lowest land value, i.e. buildings).
    - The 8 tiles in `m-eurosat_samples.png` also match.

    The fix is the errata table `CLASS_NAME_FIXES` in `scratch/explore.py`, next to `BAND_ORDER_FIXES` and under the same rule: upstream data errata only, with the evidence cited in a comment. It is a pure reordering of geobench's own name strings. `inspect_task` and both grids apply it; the card records `"class_names_corrected": true`, and its notes give the label order. **Accuracy was never affected** (the integers are consistent across splits); this only fixes the human-readable names (per-class metrics, confusion matrices, the DatasetCard, lessons in memory). **`train.py`, and any code that maps label integers to names, must use the corrected list, not `task.label_type.class_names`.** Not yet reported upstream.
15. **Native band resolutions for corrected bands come from the Sentinel-2 table.** The stored `spatial_resolution` metadata belongs to the mislabelled names, so when band names are corrected, `explore.py` takes `native_band_resolution_m` from the standard Sentinel-2 table instead (10 m: B02 B03 B04 B08; 20 m: B05 B06 B07 B8A B11 B12; 60 m: B01 B09 B10). Names outside that table (e.g. brick-kiln's `TCI_R/G/B`) get `null`. It doesn't matter for training: every eurosat band is stored at 64×64.
16. **B10 (cirrus) is almost empty in eurosat** (mean 12, std 4.8, versus thousands for other bands). After per-band standardisation it's mostly noise. It's a cheap candidate to drop, and the agent may find that on its own.
17. **pv4ger-seg has almost no panel-free tiles** (6 of 3,000). A model trained on it rarely sees an empty roof, so it may over-predict panels on images without any. Validation has the same bias, so val IoU won't reveal this.
18. **`download_tasks.py` zip-slip guard was POSIX-only.** It compared against `str(dest) + "/"`, so on Windows (paths with `\`) every zip member was refused, starting with `m-eurosat/`. It's fixed with `Path.is_relative_to`, which keeps the protection on any OS.
20. **SSL4EO weights always see 13 Sentinel-2 bands (engine, since M4).** With `weights="ssl4eo_s2"` the engine reorders the task's bands BY NAME to TorchGeo's order (B01..B08, B8A, B09..B12), divides by 10000 as in pretraining, zero-fills bands the task lacks (m-brick-kiln: B06, B09, B10) and leaves out non-Sentinel-2 channels (brick-kiln's TCI_R/G/B). The card's notes and each run's `metrics.json` (`zero_filled_bands`, `dropped_bands`) say so. imagenet/none weights use the selected bands with the train z-score instead.
21. **Mixed precision is fp16** (`torch.autocast("cuda", dtype=torch.float16)` + `GradScaler`): the SageMaker Tesla T4 has no bf16.
19. **Local CPU runtime:** `explore.py` on both tasks takes ~3m45s; data on disk is 455 MB (eurosat) + 1.5 GB (pv4ger-seg) after unzip.

---

## 6. Satellite imagery primer (for us, and for the slides)

### What a "band" is
A normal photo records 3 colours: red, green and blue light. A satellite sensor records **many narrow slices of the light spectrum**, including light humans can't see. Each slice is a **band**: one grayscale image of how much light of that wavelength the ground reflected. Stack them and every pixel becomes a vector of reflectances, a "spectral signature". Vegetation, water, soil and concrete each have characteristic signatures, which is why extra bands help.

### Sentinel-2 (m-eurosat, m-brick-kiln, and other tasks)
ESA's Sentinel-2 satellites carry the MSI sensor with 13 bands:

| Band | Name | Wavelength | Native res. | What it's good for |
|---|---|---|---|---|
| B01 | Coastal aerosol | 443 nm | 60 m | Haze and aerosols; little ground detail |
| B02 | Blue | 490 nm | 10 m | True colour; water, atmosphere |
| B03 | Green | 560 nm | 10 m | True colour; vegetation vigour |
| B04 | Red | 665 nm | 10 m | True colour; chlorophyll absorbs red, so vegetation looks dark here |
| B05 | Red edge 1 | 705 nm | 20 m | Plant stress and chlorophyll content |
| B06 | Red edge 2 | 740 nm | 20 m | Same, deeper into the red edge |
| B07 | Red edge 3 | 783 nm | 20 m | Same |
| B08 | NIR | 842 nm | 10 m | Healthy leaves reflect near-infrared strongly; key for vegetation (NDVI) |
| B8A | Narrow NIR | 865 nm | 20 m | Cleaner NIR, less water-vapour contamination |
| B09 | Water vapour | 945 nm | 60 m | Atmospheric water; mostly for correction |
| B10 | SWIR cirrus | 1375 nm | 60 m | Detects thin high clouds; very low values over land |
| B11 | SWIR 1 | 1610 nm | 20 m | Moisture in soil and plants; snow vs. cloud |
| B12 | SWIR 2 | 2190 nm | 20 m | Moisture, minerals, burned areas |

Pixel values are reflectance scaled by 10,000, so roughly 0–10,000 (bright clouds or snow can exceed it). They're stored as integers. "Native res." is the sensor's ground size per pixel; the dataset may have resampled everything to 10 m.

The "red edge" is the steep jump in plant reflectance between red (absorbed) and NIR (reflected). Its exact position shifts with plant health, which is why Sentinel-2 has three bands there.

### Aerial RGB (m-pv4ger, m-pv4ger-seg)
These come from a camera on a plane, not a satellite: 3 bands (red, green, blue) at **0.1 m per pixel**. That's 100× finer than Sentinel-2's 10 m, which is why individual rooftop solar panels are visible. Values are ordinary 0–255 photo intensities. These look like normal photos, so ImageNet-pretrained networks transfer well to them.

### Why raw pixel values can't go straight into a pretrained network
- **Scale mismatch.** Pretrained networks learned from inputs normalised to roughly mean 0 and std 1, using ImageNet statistics on 0–1 images. Raw Sentinel-2 values are in the thousands. Fed in directly, early activations explode, and batch-norm statistics and pretrained filters become meaningless. Training is unstable or doesn't learn.
- **Bands live on different scales.** B10 (cirrus) is often near 0 over land while B08 (NIR) is in the thousands. In brick-kiln, three channels are 0–255 next to reflectance bands up to ~10,000. Without per-band standardisation, the largest-valued bands dominate the gradients and the small ones are effectively ignored. That's why we compute mean/std **per band**, not one global number.
- **Channel count and meaning.** ImageNet models expect exactly 3 channels in RGB order. With 13 bands, the first convolution must be adapted (for example, copy the RGB weights to B04/B03/B02 and initialise the rest), or use weights pretrained on Sentinel-2. Even for 3 channels, the order must match what the model learned (RGB, not BGR).
- **Integer dtypes.** int16 and uint8 arrays must be cast to float32 first.

### Why normalisation statistics must come from the train split only
Mean and std are numbers we **learn from data**, exactly like model weights. If we compute them using validation or test images, information about those held-out images leaks into the preprocessing. The model is then evaluated on data that partly shaped its inputs. That makes scores optimistic and breaks the purpose of held-out data: estimating performance on images we have never seen. In production, the stats are frozen from training data and applied unchanged to new images, so evaluation must mimic that. Using test data here would also break the hackathon's strict rule: the test split is touched once, at the end, with the final model.

---

## 7. How this was tested (before touching real data)

The workspace that wrote these scripts couldn't reach Hugging Face, so the real data was **not** downloaded there. Instead:
- The download mechanism was verified against geobench 1.1.0's source and the live Hugging Face file listing.
- Three fake tasks were built **with geobench's own writer**, so the on-disk format is identical:
  - a 13-band Sentinel-2 "m-eurosat" with the upstream mislabelled band names, class imbalance, an exact duplicate, a train/val leak and nodata corners;
  - a binary aerial "m-pv4ger-seg";
  - a 4-band, 7-class segmentation task with an ignore value 255 and a class that never appears.
- The test-split files were **deliberately corrupted**, so any read of the test split would crash. None did.
- Every planted problem was detected and reported in `notes`.
- The streaming mean/std was checked against a direct NumPy computation over all pixels: max difference < 1e-6.
- `_open_split(task, "test")` raises `PermissionError`.

## 8. Next steps
1. Run the quick start on SageMaker and paste the `explore.py` output here (fill the _TO FILL_ sections).
2. Look at `docs/figures/*_samples.png`. Do eurosat tiles look like real landscapes, and do the panel masks sit on panels?
3. Commit `scratch/`, `stats/`, `docs/` to `serena/data` and push, so `train.py` can read `stats/<task>.json`.
4. Re-run `explore.py` on our other two tasks once they're downloaded (no code changes needed).
