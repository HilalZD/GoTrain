"""Model menu for the real engine: any band count, three weight options, three task types.

build_model(spec) returns a torch module plus the per-band normalisation it expects.

Weights:
  imagenet   timm (classification/multilabel) or segmentation_models_pytorch (U-Net) weights
             pretrained on ImageNet RGB. With C != 3 input bands, the first convolution is
             adapted by the library (timm `adapt_input_conv`, smp `patch_first_conv`): the RGB
             filters are repeated across the C channels and rescaled by 3/C, so the layer's
             output scale stays the same. Inputs are z-scored with stats/<task>.json.
  ssl4eo_s2  TorchGeo's Sentinel-2 SSL4EO MoCo weights (resnet18 / resnet50, 13 channels).
             The first conv's filters are picked BY BAND NAME for the bands we feed, so any
             subset or order works. Sentinel-2 bands are scaled /10000, exactly as in
             pretraining; non-Sentinel-2 bands (e.g. brick-kiln's 8-bit TCI channels) get the
             mean pretrained filter and a z-score. For segmentation the weights initialise the
             U-Net's ResNet encoder (same layer names as timm's ResNet).
  none       random initialisation, z-scored inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gotrain.engine.data import S2_NATIVE_RES_M

SSL4EO_BACKBONES = ("resnet18", "resnet50")
UNET_STRIDE = 32  # a 5-level U-Net halves the input 5 times


@dataclass
class ModelSpec:
    task_type: str  # classification | multilabel | segmentation
    backbone: str
    weights: str  # imagenet | ssl4eo_s2 | none
    bands: list[str]  # input channel names, in input order
    n_outputs: int  # logits per sample (or per pixel)
    band_mean: list[float]  # TRAIN z-score stats for these bands (same order)
    band_std: list[float]


def geobench_name(torchgeo_band: str) -> str:
    """TorchGeo 'B1' / 'B8a' -> GEO-Bench 'B01' / 'B8A'."""
    return f"B{torchgeo_band[1:].upper().zfill(2)}"


def _ssl4eo_weights(backbone: str) -> Any:
    from torchgeo.models import ResNet18_Weights, ResNet50_Weights

    enum = {"resnet18": ResNet18_Weights, "resnet50": ResNet50_Weights}[backbone]
    return enum.SENTINEL2_ALL_MOCO


def check_spec(spec: ModelSpec) -> str | None:
    """A human-readable reason this spec cannot be built, or None."""
    if spec.weights == "ssl4eo_s2":
        if spec.backbone not in SSL4EO_BACKBONES:
            return (
                f"weights 'ssl4eo_s2' exist only for backbones {list(SSL4EO_BACKBONES)}, "
                f"not {spec.backbone!r}; use one of those or weights 'imagenet'"
            )
        if not any(b in S2_NATIVE_RES_M for b in spec.bands):
            return "weights 'ssl4eo_s2' need Sentinel-2 bands (B01..B12, B8A); none selected"
    if spec.task_type == "segmentation":
        import segmentation_models_pytorch as smp

        names = smp.encoders.get_encoder_names()
        if spec.backbone not in names and not spec.backbone.startswith("tu-"):
            return (
                f"unknown U-Net encoder {spec.backbone!r}; try resnet18, resnet34, resnet50, "
                "efficientnet-b0 or tu-convnext_tiny"
            )
    else:
        import timm

        if not timm.is_model(spec.backbone):
            return (
                f"unknown timm backbone {spec.backbone!r}; try resnet18, resnet50, "
                "efficientnet_b0 or convnext_tiny"
            )
    return None


def input_normalisation(spec: ModelSpec) -> tuple[np.ndarray, np.ndarray]:
    """Per-band (mean, std) to apply as (x - mean) / std, matching the chosen weights."""
    mean = np.asarray(spec.band_mean, np.float32).copy()
    std = np.asarray(spec.band_std, np.float32).copy()
    if spec.weights == "ssl4eo_s2":
        pretrained = {geobench_name(b) for b in _ssl4eo_weights(spec.backbone).meta["bands"]}
        for i, band in enumerate(spec.bands):
            if band in pretrained:
                mean[i], std[i] = 0.0, 10000.0
    return mean, np.where(std < 1e-6, 1.0, std).astype(np.float32)


def _ssl4eo_state(spec: ModelSpec) -> dict:
    """SSL4EO state dict with conv1 reduced to our bands (by name), fc dropped."""
    import torch

    weights = _ssl4eo_weights(spec.backbone)
    state = weights.get_state_dict(progress=True, check_hash=True, weights_only=True)
    order = [geobench_name(b) for b in weights.meta["bands"]]
    w = state["conv1.weight"]  # [64, 13, 7, 7]
    mean_filter = w.mean(dim=1)
    cols = [w[:, order.index(b)] if b in order else mean_filter for b in spec.bands]
    state["conv1.weight"] = torch.stack(cols, dim=1).contiguous()
    return {k: v for k, v in state.items() if not k.startswith("fc.")}


def build_model(spec: ModelSpec) -> Any:
    problem = check_spec(spec)
    if problem:
        raise ValueError(problem)
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
        encoder_name=spec.backbone,
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
