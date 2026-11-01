# Data audit: m-eurosat (GoTrain errata applied)

- Task type: classification; 300 sampled TRAIN tiles; duplicates checked on 2000 train / 1000 val tiles. The test split is never read.
- Bands: B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A
- Classes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, Sea and Lake
- Findings: 2 info

| severity | check | finding | evidence |
|---|---|---|---|
| info | shortcuts | A model must clearly beat always predicting the majority class. | `{"majority_class_baseline": 0.1, "metric": "accuracy", "predict": "Annual Crop"}` |
| info | shortcuts | Open water is flat, so 'Sea and Lake' having most flat tiles is expected. | `{"measure": "flat (low-contrast) tiles", "highest": ["Sea and Lake", 0.7647], "lowest": ["Annual Crop", 0.0], "per_class": {"Annual Crop": 0.0, "Forest": 0.0323, "Herbaceous Vegetation": 0.0, "Highway": 0.0, "Industrial Buildings": 0.0, "Pasture": 0.0, "Permanent Crop": 0.0, "Residential Buildings": 0.0, "River": 0.0, "Sea and Lake": 0.7647}}` |
