# Data audit: m-eurosat (raw GEO-Bench metadata)

- Task type: classification; 300 sampled TRAIN tiles; duplicates checked on 2000 train / 1000 val tiles. The test split is never read.
- Bands: 01 - Coastal aerosol, 02 - Blue, 03 - Green, 04 - Red, 05 - Vegetation Red Edge, 06 - Vegetation Red Edge, 07 - Vegetation Red Edge, 08 - NIR, 08A - Vegetation Red Edge, 09 - Water vapour, 10 - SWIR - Cirrus, 11 - SWIR, 12 - SWIR
- Classes: Industrial Buildings, Residential Buildings, Annual Crop, Permanent Crop, River, Sea and Lake, Herbaceous Vegetation, Highway, Pasture, Forest
- Findings: 7 error, 1 info, 1 warning

| severity | check | finding | evidence |
|---|---|---|---|
| error | bands | The channel named '08A - Vegetation Red Edge' doesn't move with B08 (NIR) as B8A should. Channel '12 - SWIR' correlates r=0.97 with B08: it behaves like B8A. The band names are probably out of order. | `{"channel": "08A - Vegetation Red Edge", "r_with_B08": 0.686, "expected": ">= 0.9"}` |
| error | bands | The two channels named SWIR B11/B12 don't move together as SWIR bands do. The band names are probably out of order. | `{"channels": ["11 - SWIR", "12 - SWIR"], "r": 0.327, "expected": ">= 0.8"}` |
| error | bands | The channel named '10 - SWIR - Cirrus' should be B10 (cirrus), which is near zero over land, but it isn't. Near-zero channel(s) '09 - Water vapour' look like B10 instead. | `{"channel": "10 - SWIR - Cirrus", "p99": 4090.0, "brightest_p99": 5128.0}` |
| error | classes | 'Residential Buildings' contradicts its spectra: built-up classes are not dense vegetation (NDVI < 0.6). The data look like: 'Forest'. | `{"class": "Residential Buildings", "kind_from_name": "built", "mean_ndvi": 0.696, "mean_ndwi": -0.557}` |
| error | classes | 'Sea and Lake' contradicts its spectra: water has low NDVI; this class is strongly vegetated. The data look like: 'Forest'. | `{"class": "Sea and Lake", "kind_from_name": "water", "mean_ndvi": 0.641, "mean_ndwi": -0.534}` |
| error | classes | 'Forest' contradicts its spectra: forest/tree classes are dense vegetation (NDVI > 0.2). The data look like: 'River', 'Sea and Lake'. | `{"class": "Forest", "kind_from_name": "forest", "mean_ndvi": -0.233, "mean_ndwi": 0.457}` |
| error | classes | Several class names contradict their data: the class-name list is probably out of order relative to the label integers. Check a few tiles per label. | `{"contradicting_classes": 3}` |
| warning | shortcuts | Flat (low-contrast) tiles are much more common in 'Forest' than in 'Industrial Buildings': a model can score by spotting them instead of the content. Report per-class recall of 'Industrial Buildings' and look at the error grid. | `{"measure": "flat (low-contrast) tiles", "highest": ["Forest", 0.7647], "lowest": ["Industrial Buildings", 0.0], "per_class": {"Industrial Buildings": 0.0, "Residential Buildings": 0.0323, "Annual Crop": 0.0, "Permanent Crop": 0.0, "River": 0.0, "Sea and Lake": 0.0, "Herbaceous Vegetation": 0.0, "Highway": 0.0, "Pasture": 0.0, "Forest": 0.7647}}` |
| info | shortcuts | A model must clearly beat always predicting the majority class. | `{"majority_class_baseline": 0.1, "metric": "accuracy", "predict": "Industrial Buildings"}` |
