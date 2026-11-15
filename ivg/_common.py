"""Internal utilities: input validation and visibility criteria."""

from __future__ import annotations

import numpy as np

CRITERIA = ("horizontal", "natural")

_ALIASES = {
    "horizontal": "horizontal",
    "hvg": "horizontal",
    "ihvg": "horizontal",
    "natural": "natural",
    "vg": "natural",
    "ivg": "natural",
}


def check_criterion(criterion: str) -> str:
    """Normalize the name of the visibility criterion.

    Accepts ``"horizontal"`` (aliases ``"hvg"``, ``"ihvg"``) and ``"natural"``
    (aliases ``"vg"``, ``"ivg"``).
    """
    key = str(criterion).lower()
    if key not in _ALIASES:
        raise ValueError(
            f"criterion must be 'horizontal' or 'natural' (got {criterion!r})."
        )
    return _ALIASES[key]


def as_image(img, min_size: int = 1) -> np.ndarray:
    """Validate a 2D grayscale image and return it as float64.

    The image is converted to ``float64`` (not to an integer type) so that
    real-valued images, e.g. normalized to [0, 1], are not truncated. 8/16-bit
    integers are represented exactly in float64, so comparisons stay exact.
    """
    a = np.asarray(img)
    if a.ndim != 2:
        hint = (
            " Convert it to grayscale first (e.g. ivg.load_gray or "
            "cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))."
            if a.ndim == 3
            else ""
        )
        raise ValueError(f"The image must be 2D (H, W); got shape {a.shape}.{hint}")
    if not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):
        raise TypeError(f"The image must be numeric (got dtype {a.dtype}).")
    if np.iscomplexobj(a):
        raise TypeError("The image cannot be complex-valued.")
    a = a.astype(np.float64, copy=False)
    if not np.all(np.isfinite(a)):
        raise ValueError("The image contains NaN or infinite values.")
    H, W = a.shape
    if H < min_size or W < min_size:
        raise ValueError(f"The image must be at least {min_size}x{min_size} pixels.")
    return a


def load_gray(path) -> np.ndarray:
    """Load an image from disk as grayscale (``uint8``/``uint16``).

    Unlike ``cv2.imread``, raises a clear error if the file does not exist or
    cannot be read, instead of silently returning ``None``.
    """
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image file not found: {p}")
    try:
        import cv2

        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    except ImportError:
        pass
    from PIL import Image

    with Image.open(p) as im:
        return np.asarray(im.convert("L"))
