"""Training rules the real engine enforces, checked up front so the agent never wastes a trial.

The real engine (gotrain/engine/train.py and models.py, Serena's) defines these as LOSSES,
BINARY_SEG_LOSSES and UNET_STRIDE; tests/test_engine_rules.py fails if this copy drifts.
The tools check them before training and the fake engine fails on them like the real one.
"""

from gotrain.contracts import DatasetCard, TrainConfig
from gotrain.engine.combos import check_combo

LOSSES_BY_TASK_TYPE: dict[str, tuple[str, ...]] = {
    "classification": ("ce", "focal", "bce"),
    "multilabel": ("bce", "focal"),
    "segmentation": ("ce", "dice_ce", "focal", "bce", "dice_bce"),
}
BINARY_SEG_LOSSES = ("bce", "dice_bce")  # one logit per pixel: needs exactly 2 classes
SEG_IMG_MULTIPLE = 32  # a 5-level U-Net halves the input 5 times


def allowed_losses(card: DatasetCard) -> list[str]:
    losses = LOSSES_BY_TASK_TYPE[card.task_type]
    if card.task_type == "segmentation" and card.n_classes != 2:
        losses = tuple(loss for loss in losses if loss not in BINARY_SEG_LOSSES)
    return list(losses)


def rules_for(card: DatasetCard) -> dict:
    """The rules for this task, as inspect_dataset shows them to the agent."""
    rules: dict = {"allowed_losses": allowed_losses(card)}
    if card.task_type == "segmentation":
        rules["binary_only_losses"] = list(BINARY_SEG_LOSSES)
        rules["img_size_multiple_of"] = SEG_IMG_MULTIPLE
    return rules


def check_task_rules(card: DatasetCard, config: TrainConfig) -> str | None:
    """None if the config obeys the engine's rules for this task, else a fix-it message."""
    allowed = allowed_losses(card)
    if config.loss not in allowed:
        if card.task_type == "segmentation" and config.loss in BINARY_SEG_LOSSES:
            why = f"it is for 2-class masks and {card.task} has {card.n_classes} classes"
        else:
            why = f"it does not fit a {card.task_type} task"
        return f"loss={config.loss!r} can't be used: {why}. Allowed for {card.task}: {allowed}."
    if card.task_type == "segmentation" and config.img_size % SEG_IMG_MULTIPLE:
        lower = config.img_size // SEG_IMG_MULTIPLE * SEG_IMG_MULTIPLE
        options = [s for s in (lower, lower + SEG_IMG_MULTIPLE) if s >= SEG_IMG_MULTIPLE]
        return (
            f"img_size={config.img_size} is not a multiple of {SEG_IMG_MULTIPLE}, which "
            f"segmentation models need. Use e.g. {' or '.join(map(str, options))}."
        )
    return None


def check_config(card: DatasetCard, config: TrainConfig) -> str | None:
    """Everything the real engine would reject before training: combos, then task rules."""
    return check_combo(card, config.backbone, config.weights) or check_task_rules(card, config)
