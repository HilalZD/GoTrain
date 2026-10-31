"""Valid backbone + weights combinations, as confirmed for the real engine.

The single source of truth: the tools reject other combos before training, the fake
engine fails on them like the real one, and prompts/tune.md lists the same table
(tests/test_combos.py keeps the prompt in sync).
"""

from gotrain.contracts import DatasetCard

ALL_BACKBONES = ("resnet18", "resnet34", "resnet50", "convnext_tiny")
VALID_COMBOS: dict[str, tuple[str, ...]] = {
    "imagenet": ALL_BACKBONES,
    "none": ALL_BACKBONES,
    "ssl4eo_s2": ("resnet18", "resnet50"),
}
SENTINEL2_ONLY_WEIGHTS = {"ssl4eo_s2"}


def is_sentinel2(card: DatasetCard) -> bool:
    """True for Sentinel-2 sensors, however the card spells it ('sentinel-2', 'Sentinel2'...)."""
    return card.sensor.lower().replace("-", "").replace("_", "").replace(" ", "") == "sentinel2"


def valid_combos_for(card: DatasetCard) -> dict[str, list[str]]:
    """The weights -> backbones options that are valid for this task."""
    return {
        weights: list(backbones)
        for weights, backbones in VALID_COMBOS.items()
        if weights not in SENTINEL2_ONLY_WEIGHTS or is_sentinel2(card)
    }


def describe_combos(card: DatasetCard) -> str:
    options = "; ".join(f"weights={w!r}: {', '.join(b)}" for w, b in valid_combos_for(card).items())
    return f"Valid backbone + weights combinations for {card.task} ({card.sensor}): {options}."


def check_combo(card: DatasetCard, backbone: str, weights: str) -> str | None:
    """None if the combo is valid for this task, else a message listing the valid options."""
    if weights in SENTINEL2_ONLY_WEIGHTS and not is_sentinel2(card):
        problem = f"weights={weights!r} only works on Sentinel-2 tasks, not {card.sensor}"
    elif backbone not in VALID_COMBOS.get(weights, ()):
        problem = f"backbone={backbone!r} is not available with weights={weights!r}"
    else:
        return None
    return f"{problem}. {describe_combos(card)}"
