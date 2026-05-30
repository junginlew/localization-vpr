"""aidall_apriltag — a friendly Python wrapper around the official AprilTag library.

Quick start
-----------
    import aidall_apriltag as at

    detector = at.Detector(family="tag36h11")
    detections = detector.detect(gray_or_color_image)  # NumPy array

    for d in detections:
        print(d.id, d.center)

Pose estimation
---------------
    detector = at.Detector(
        family="tag36h11",
        tag_size=0.16,                       # meters
        camera=at.CameraParams(fx, fy, cx, cy),
    )
    for d in detector.detect(frame):
        if d.pose is not None:
            print(d.id, d.pose.t)            # translation in meters

The detector accepts grayscale (H, W) or color (H, W, 3/4) uint8 arrays; color
images are converted to grayscale automatically.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np

from . import _core
from ._core import CameraParams, Detection, Family, Pose
from .generate import (
    add_quiet_zone,
    border_size_to_total_mm,
    generate,
    save_pdf_sheet,
    save_png,
    save_svg,
    scale,
    to_svg,
)

__all__ = [
    "Detector",
    "CameraParams",
    "Detection",
    "Pose",
    "Family",
    "family_from_string",
    "family_size",
    # generation
    "generate",
    "add_quiet_zone",
    "scale",
    "save_png",
    "to_svg",
    "save_svg",
    "save_pdf_sheet",
    "border_size_to_total_mm",
    "__version__",
]


def family_size(family: FamilyLike) -> int:
    """Number of distinct tag IDs available in a family."""
    fam = family_from_string(family) if isinstance(family, str) else family
    return _core.family_size(fam)


try:
    from ._version import __version__
except ImportError:  # pragma: no cover - populated at build time
    __version__ = "0.0.0+unknown"

FamilyLike = Union[Family, str]


def family_from_string(name: str) -> Family:
    """Resolve a family name like ``"tag36h11"`` to a :class:`Family`."""
    fam = _core.family_from_string(name)
    if fam is None:
        raise ValueError(f"unknown AprilTag family: {name!r}")
    return fam


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Coerce an image to a contiguous 2-D uint8 grayscale array."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        # Rec. 601 luma; matches what OpenCV's BGR2GRAY produces closely enough.
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        # Assume BGR ordering (OpenCV default); weights are symmetric enough
        # that RGB input only shifts intensities slightly.
        b, g, r = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        arr = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
    elif arr.ndim != 2:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


class Detector:
    """High-level AprilTag detector.

    Parameters mirror :class:`aidall_apriltag._core.DetectorOptions` but accept
    a family name string for convenience.
    """

    def __init__(
        self,
        family: FamilyLike = Family.Tag36h11,
        *,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: bool = True,
        nthreads: int = 1,
        tag_size: float = 0.0,
        camera: Optional[CameraParams] = None,
    ) -> None:
        options = _core.DetectorOptions()
        options.family = (
            family_from_string(family) if isinstance(family, str) else family
        )
        options.quad_decimate = quad_decimate
        options.quad_sigma = quad_sigma
        options.refine_edges = refine_edges
        options.nthreads = nthreads
        options.tag_size = tag_size
        if camera is not None:
            options.camera = camera
        self._detector = _core.Detector(options)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect tags in a grayscale or color uint8 image (NumPy array)."""
        return self._detector.detect(_to_gray(image))

    def set_camera(self, camera: CameraParams) -> None:
        self._detector.set_camera(camera)

    def set_tag_size(self, meters: float) -> None:
        self._detector.set_tag_size(meters)
