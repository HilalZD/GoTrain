# GoPilot-ready export

**Pitch:** GoTrain does not stop at a leaderboard: every model the agent trains is exported
the moment it is registered as a documented, callable tool that a geospatial assistant such as
GoPilot can use, so the story runs from raw data to audit to model to production.

## What it is

- **Auditor tool** (`audit_dataset`): the agent must call the data auditor before its first
  trial (`AuditGate`; real engine, once `gotrain/engine/audit.py` is merged). Results are cached
  in `registry/<task>/data_audit.json`; it is free and never charged to the GPU budget.
- **Bundle** (`gotrain/gopilot.py`): `registry/<task>/<version>/gopilot/bundle.json` +
  `model_card.md`. It holds the checkpoint list and the ensemble/TTA choice that won on
  VALIDATION, the sensor, bands in order, image size, the dataset card's corrected class names,
  validation metrics, the test score and CI once `final_test` has run, calibration info if the
  engine wrote any, and a version hash. Predictions call the engine's `predict()` (contract v2).
  No TorchScript/ONNX, no tiling.
- **Automatic:** the driver exports after every tune registration and again after finalise (this
  adds the test score). The journal names the tool, e.g. `detect_brick_kilns`. Manual:
  `export_model(task, version)`.
- **MCP server** (`gotrain/mcp/models_server.py`): `list_models` plus one tool per bundle:
  `classify_land_cover_eurosat`, `detect_brick_kilns`, `segment_solar_panels`,
  `map_land_cover_chesapeake` (descriptions generated from the card).
  - Input: path to a GeoTIFF or `.npy` (C x H x W, bands in the card's order).
  - Classification: label, probabilities, confidence.
  - Segmentation: mask GeoTIFF (georeferenced when the input is), GeoJSON polygons of the target
    classes (capped at 200, with the total count), pixels and area (m2 for projected CRS) per
    class, confidence.
  - `needs_review` is passed on when `predict()` returns it. A wrong band count returns a clear
    `{"error": ..., "hint": ...}`.

## Run it

```bash
# server (stdio), e.g. from an MCP client config
GOTRAIN_HOME=~/gotrain_home python -m gotrain.mcp.models_server

# demo: a Strands agent (Bedrock, CPU only) asks "Are there brick kilns in this tile?"
GOTRAIN_ENGINE=real GOTRAIN_HOME=~/gotrain_home python scripts/demo_gopilot.py --task m-brick-kiln
```

The demo picks a VALIDATION tile from the data cache (`val_x.npy`/`val_y.npy`; never the test
split), starts the server as a subprocess, and exits non-zero unless the agent called the tool.
On a dev machine with the fake engine it uses a synthetic tile. `GOTRAIN_MCP_OUT` sets where
mask GeoTIFFs are written (default: the system temp folder).
