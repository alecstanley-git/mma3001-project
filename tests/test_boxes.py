"""Tests of bounding box geometry."""

import numpy as np
import pytest

from packaging_defects.boxes import (
    centre_to_corner,
    corner_to_centre,
    intersection_over_union,
    mirror_boxes,
    non_maximum_suppression,
)


def test_centre_to_corner_known_value():
    assert np.allclose(centre_to_corner([[0.5, 0.5, 0.2, 0.4]]), [[0.4, 0.3, 0.6, 0.7]])


def test_format_conversion_round_trip():
    boxes = np.random.default_rng(0).uniform(0.1, 0.4, (20, 4))
    assert np.allclose(corner_to_centre(centre_to_corner(boxes)), boxes)


def test_overlap_of_identical_boxes_is_one():
    box = [[0.1, 0.1, 0.5, 0.5]]
    assert intersection_over_union(box, box)[0, 0] == pytest.approx(1.0)


def test_overlap_of_separate_boxes_is_zero():
    assert intersection_over_union([[0.0, 0.0, 0.2, 0.2]], [[0.5, 0.5, 0.9, 0.9]])[0, 0] == 0.0


def test_overlap_known_value():
    # Two unit squares offset by half a side share 0.5 and cover 1.5 together.
    overlap = intersection_over_union([[0.0, 0.0, 1.0, 1.0]], [[0.5, 0.0, 1.5, 1.0]])
    assert overlap[0, 0] == pytest.approx(1.0 / 3.0)


def test_overlap_is_unchanged_by_stretching_the_axes():
    first, second = np.array([[0.1, 0.2, 0.5, 0.6]]), np.array([[0.3, 0.3, 0.7, 0.9]])
    stretch = np.array([720.0, 540.0, 720.0, 540.0])
    assert intersection_over_union(first, second)[0, 0] == pytest.approx(
        intersection_over_union(first * stretch, second * stretch)[0, 0]
    )


def test_overlap_returns_one_value_per_pair():
    assert intersection_over_union(np.zeros((3, 4)), np.zeros((5, 4))).shape == (3, 5)


def test_suppression_removes_repeats_and_keeps_separate_boxes():
    boxes = np.array(
        [[0.10, 0.10, 0.30, 0.30], [0.11, 0.10, 0.31, 0.30], [0.60, 0.60, 0.80, 0.80]]
    )
    kept = non_maximum_suppression(boxes, np.array([0.6, 0.9, 0.5]), overlap_threshold=0.5)
    assert kept.tolist() == [1, 2]


def test_suppression_of_nothing_returns_nothing():
    assert len(non_maximum_suppression(np.zeros((0, 4)), np.zeros(0))) == 0


def test_suppression_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        non_maximum_suppression(np.zeros((2, 4)), np.zeros(3))


def test_mirror_boxes_moves_centres_only():
    mirrored = mirror_boxes([[0.2, 0.3, 0.1, 0.4]], left_right=True, up_down=True)
    assert np.allclose(mirrored, [[0.8, 0.7, 0.1, 0.4]])
