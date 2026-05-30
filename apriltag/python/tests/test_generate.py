"""Tests for tag generation, including a generate -> detect round-trip."""

import aidall_apriltag as at
import numpy as np
import pytest


def test_generate_shape_and_values():
    grid = at.generate("tag36h11", 0)
    assert grid.ndim == 2
    assert grid.dtype == np.uint8
    # rendered grid is total_width square
    assert grid.shape[0] == grid.shape[1]
    # only black/white
    assert set(np.unique(grid)).issubset({0, 255})


def test_quiet_zone_and_scale():
    grid = at.generate("tag36h11", 0)
    padded = at.add_quiet_zone(grid, cells=2)
    assert padded.shape[0] == grid.shape[0] + 4
    assert (padded[0, :] == 255).all()  # border is white

    big = at.scale(grid, pixels_per_cell=5)
    assert big.shape == (grid.shape[0] * 5, grid.shape[1] * 5)


@pytest.mark.parametrize("tag_id", [0, 7, 42])
def test_generate_then_detect_roundtrip(tag_id):
    """A generated tag must be detected with the same id."""
    grid = at.add_quiet_zone(at.generate("tag36h11", tag_id), cells=3)
    pixels = at.scale(grid, pixels_per_cell=10)
    detector = at.Detector(family="tag36h11")
    detections = detector.detect(pixels)
    assert len(detections) == 1
    assert detections[0].id == tag_id


def test_invalid_id_raises():
    with pytest.raises((IndexError, ValueError, OverflowError, RuntimeError)):
        at.generate("tag36h11", 10_000_000)


def test_to_svg_is_wellformed():
    svg = at.to_svg("tag36h11", 0, tag_size_mm=50.0)
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "mm" in svg  # exact physical sizing
