"""Model menu for the real engine: any band count, three weight options, three task types.

Which backbone + weights combinations are allowed is decided by gotrain/engine/combos.py
(Hilal's file, the single source of truth); this module only builds them.

Weights:
  imagenet   timm (classification/multilabel) or segmentation_models_pytorch (U-Net) weights
             pretrained on ImageNet RGB. With C != 3 input bands, the first convolution is
             adapted by the library (timm `adapt_input_conv`, smp `patch_first_conv`): the RGB
             filters are repeated across the C channels and rescaled by 3/C, so the layer's
             output scale stays the same. Inputs are z-scored with stats/<task>.json.
  ssl4eo_s2  TorchGeo's Sentinel-2 SSL4EO MoCo weights (ResNet18/ResNet50_Weights.
             SENTINEL2_ALL_MOCO), verified in torchgeo 0.10.1: 13 input channels in the order
             B1..B8, B8a, B9..B12, pretrained on input / 10000. The model always receives those
             13 channels in that order: the task's bands are reordered BY NAME, bands the task
             lacks (m-brick-kiln: B06, B09, B10) are zero-filled, and non-Sentinel-2 bands
             (brick-kiln's TCI_R/G/B) are left out. For segmentation the same weights
             initialise the U-Net's ResNet encoder (same layer names as timm's ResNet).
  none       random initialisation, z-scored inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gotrain.engine.data import SSL4EO_S2_BANDS

UNET_STRIDE = 32  # a 5-level U-Net halves the input 5 times
SMP_ENCODER = {"convnext_tiny": "tu-convnext_tiny"}  # smp reaches convnext through timm


@dataclass
class ModelSpec:
    task_type: str  # classification | multilabel | segmentation
    backbone: str
    weights: str  # imagenet | ssl4eo_s2 | none
    bands: list[str]  # model input channels, in input order
    n_outputs: int  # logits per sample (or per pixel)
    band_mean: list[float]  # TRAIN z-score stats for these bands (same order)
    band_std: list[float]
    zero_fill: list[str] = field(default_factory=list)  # input channels fed as zeros


def ssl4eo_inputs(selected: list[str]) -> tuple[list[str], list[str], list[str]]:
    """(13 model input bands, zero-filled bands, selected bands left out) for ssl4eo_s2."""
    zero = [b for b in SSL4EO_S2_BANDS if b not in selected]
    dropped = [b for b in selected if b not in SSL4EO_S2_BANDS]
    return list(SSL4EO_S2_BANDS), zero, dropped


def _ssl4eo_weights(backbone: str) -> Any:
    from torchgeo.models import ResNet18_Weights, ResNet50_Weights

    enum = {"resnet18": ResNet18_Weights, "resnet50": ResNet50_Weights}[backbone]
    return enum.SENTINEL2_ALL_MOCO


def input_normalisation(spec: ModelSpec) -> tuple[np.ndarray, np.ndarray]:
    """Per-band (mean, std) to apply as (x - mean) / std, matching the chosen weights."""
    if spec.weights == "ssl4eo_s2":  # exactly how the weights were pretrained
        n = len(spec.bands)
        return np.zeros(n, np.float32), np.full(n, 10000.0, np.float32)
    mean = np.asarray(spec.band_mean, np.float32)
    std = np.asarray(spec.band_std, np.float32)
    return mean, np.where(std < 1e-6, 1.0, std).astype(np.float32)


def _ssl4eo_state(spec: ModelSpec) -> dict:
    """SSL4EO state dict (13-channel conv1) without the classifier head."""
    weights = _ssl4eo_weights(spec.backbone)
    order = [f"B{b[1:].upper().zfill(2)}" for b in weights.meta["bands"]]  # 'B8a' -> 'B8A'
    if order != list(spec.bands):
        raise RuntimeError(f"SSL4EO band order {order} != model inputs {spec.bands}")
    state = weights.get_state_dict(progress=True, check_hash=True, weights_only=True)
    return {k: v for k, v in state.items() if not k.startswith("fc.")}


def build_model(spec: ModelSpec) -> Any:
    if spec.task_type == "segmentation":
        return _build_unet(spec)
    return _build_classifier(spec)


def _build_classifier(spec: ModelSpec) -> Any:
    import timm

    model = timm.create_model(
        spec.backbone,
        pretrained=spec.weights == "imagenet",
        in_chans=len(spec.bands),
        num_classes=spec.n_outputs,
    )
    if spec.weights == "ssl4eo_s2":
        missing, unexpected = model.load_state_dict(_ssl4eo_state(spec), strict=False)
        _check_loaded(missing, unexpected, allowed_missing=("fc.",))
    return model


def _build_unet(spec: ModelSpec) -> Any:
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name=SMP_ENCODER.get(spec.backbone, spec.backbone),
        encoder_weights="imagenet" if spec.weights == "imagenet" else None,
        in_channels=len(spec.bands),
        classes=spec.n_outputs,
    )
    if spec.weights == "ssl4eo_s2":
        missing, unexpected = model.encoder.load_state_dict(_ssl4eo_state(spec), strict=False)
        _check_loaded(missing, unexpected, allowed_missing=())
    return model


def _check_loaded(missing: list[str], unexpected: list[str], allowed_missing: tuple) -> None:
    bad_missing = [k for k in missing if not k.startswith(allowed_missing)]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"pretrained weights did not fit the model: missing {bad_missing[:5]}, "
            f"unexpected {unexpected[:5]}"
        )
