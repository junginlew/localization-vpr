"""Tests for the aidall_apriltag Python package."""

import aidall_apriltag as at
import numpy as np
import pytest


def test_family_resolution():
    assert at.family_from_string("tag36h11") == at.Family.Tag36h11
    with pytest.raises(ValueError):
        at.family_from_string("not-a-family")


def test_blank_image_has_no_detections():
    detector = at.Detector(family="tag36h11")
    blank = np.zeros((48, 64), dtype=np.uint8)
    assert detector.detect(blank) == []


def test_accepts_color_image():
    detector = at.Detector()
    color = np.zeros((48, 64, 3), dtype=np.uint8)
    # Should coerce to grayscale internally without raising.
    assert detector.detect(color) == []


def test_pose_disabled_without_camera():
    detector = at.Detector(family="tag36h11", tag_size=0.1)  # no camera params
    blank = np.zeros((48, 64), dtype=np.uint8)
    assert detector.detect(blank) == []
