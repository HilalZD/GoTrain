"""Prepare a diagnostic PNG for the model: at most 1024 px on the long side, under ~1 MB.

Only the diagnose tool shows images to the agent; this module never logs image bytes.
"""

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_SIDE_PX = 1024
MAX_BYTES = 1_000_000
MIN_SIDE_PX = 64
SHRINK_STEP = 0.8


class ImageError(Exception):
    """The image is missing, unreadable, or can't be made small enough."""


@dataclass(frozen=True)
class PreparedImage:
    png: bytes
    width: int
    height: int
    original_width: int
    original_height: int

    def info(self) -> dict:
        """Everything about the image except its bytes (safe to return and to log)."""
        return {
            "width": self.width,
            "height": self.height,
            "size_bytes": len(self.png),
            "original_size": [self.original_width, self.original_height],
        }


def _encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def prepare_image(
    path: Path, max_side: int = MAX_SIDE_PX, max_bytes: int = MAX_BYTES
) -> PreparedImage:
    """Load an image, downscale it to fit max_side, and shrink further until under max_bytes."""
    try:
        with Image.open(path) as source:
            source.load()
            image = source.convert("RGBA" if "A" in source.getbands() else "RGB")
    except (OSError, UnidentifiedImageError) as exc:  # missing file, not an image, truncated
        raise ImageError(f"cannot read image {path}: {exc}") from exc

    original = image.size
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    png = _encode_png(image)
    while len(png) > max_bytes:
        new_size = (int(image.width * SHRINK_STEP), int(image.height * SHRINK_STEP))
        if min(new_size) < MIN_SIDE_PX:
            raise ImageError(f"cannot fit {path} under {max_bytes} bytes")
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        png = _encode_png(image)
    return PreparedImage(png, image.width, image.height, *original)
