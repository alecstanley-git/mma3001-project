"""Tests of turning boxes into grid targets and back."""

import numpy as np
import pytest

from packaging_defects.boxes import centre_to_corner
from packaging_defects.grid import Grid, decode, encode, loss_weights

GRID = Grid(rows=9, columns=12, number_of_classes=3)


def test_outputs_per_cell():
    assert GRID.outputs_per_cell == 8


def test_encode_places_the_defect_in_the_right_cell():
    # Centre (0.30, 0.50) lies in column int(0.30 * 12) = 3 and row int(0.50 * 9) = 4.
    targets = encode([[0.30, 0.50, 0.16, 0.25]], [2], GRID)
    assert targets.shape == (9, 12, 8)
    assert targets[..., 0].sum() == 1.0
    cell = targets[4, 3]
    assert cell[0] == 1.0
    assert cell[1] == pytest.approx(0.6)  # 0.30 * 12 = 3.6
    assert cell[2] == pytest.approx(0.5)  # 0.50 * 9 = 4.5
    assert cell[3] == pytest.approx(0.4) and cell[4] == pytest.approx(0.5)  # square roots
    assert cell[5:].tolist() == [0.0, 0.0, 1.0]


def test_encode_of_no_defects_is_all_zero():
    assert not encode(np.zeros((0, 4)), [], GRID).any()


def test_encode_handles_a_centre_on_the_far_edge():
    targets = encode([[1.0, 1.0, 0.1, 0.1]], [0], GRID)
    assert targets[8, 11, 0] == 1.0


def test_large_defect_is_reported_by_every_cell_in_its_central_part():
    # A box 0.4 wide and 0.4 high centred on the image.  Its central half
    # reaches 0.1 either side of the centre, which covers the middles of
    # columns 5 and 6 in row 4 and no others.
    targets = encode([[0.5, 0.5, 0.4, 0.4]], [1], GRID)
    rows, columns = np.nonzero(targets[..., 0])
    assert list(zip(rows.tolist(), columns.tolist())) == [(4, 5), (4, 6)]
    # Both cells point at the same centre: one cell to the right of column 5's
    # left edge, and exactly on column 6's left edge.
    assert targets[4, 5, 1] == pytest.approx(1.0) and targets[4, 6, 1] == pytest.approx(0.0)
    # Decoding gives the same box twice, and the repeat is merged.
    boxes, scores, classes = decode(targets, GRID)
    assert np.allclose(boxes, [[0.3, 0.3, 0.7, 0.7]], atol=1e-6)
    assert classes.tolist() == [1]


def test_central_fraction_of_zero_uses_only_the_centre_cell():
    targets = encode([[0.5, 0.5, 0.4, 0.4]], [1], GRID, central_fraction=0.0)
    assert targets[..., 0].sum() == 1.0 and targets[4, 6, 0] == 1.0


def test_smaller_defect_keeps_a_cell_claimed_by_two_defects():
    large, small = [0.5, 0.5, 0.8, 0.8], [0.5, 0.5, 0.05, 0.05]
    for boxes, classes in (([large, small], [0, 2]), ([small, large], [2, 0])):
        centre_cell = encode(boxes, classes, GRID)[4, 6]
        assert centre_cell[3] == pytest.approx(np.sqrt(0.05))
        assert centre_cell[5:].tolist() == [0.0, 0.0, 1.0]


def test_decode_recovers_encoded_boxes():
    boxes = np.array([[0.30, 0.50, 0.16, 0.25], [0.80, 0.15, 0.10, 0.12]])
    recovered, scores, classes = decode(encode(boxes, [2, 0], GRID), GRID)
    order = np.argsort(recovered[:, 0])
    assert np.allclose(recovered[order], centre_to_corner(boxes), atol=1e-6)
    assert np.allclose(scores, 1.0)
    assert classes[order].tolist() == [2, 0]


def test_decode_ignores_unconfident_cells_and_impossible_sizes():
    outputs = np.zeros((9, 12, 8))
    outputs[2, 2] = [0.005, 0.5, 0.5, 0.3, 0.3, 1, 0, 0]  # below the minimum confidence
    outputs[5, 5] = [0.9, 0.5, 0.5, -0.3, 0.3, 1, 0, 0]  # negative width
    boxes, _, _ = decode(outputs, GRID)
    assert len(boxes) == 0


def test_decode_clips_confidence_and_boxes_into_range():
    outputs = np.zeros((9, 12, 8))
    outputs[0, 0] = [1.7, 0.1, 0.1, 0.5, 0.5, 0, 1, 0]
    boxes, scores, classes = decode(outputs, GRID)
    assert scores.tolist() == [1.0]
    assert boxes.min() >= 0.0 and boxes.max() <= 1.0
    assert classes.tolist() == [1]


def test_decode_merges_neighbouring_cells_reporting_the_same_defect():
    outputs = np.zeros((9, 12, 8))
    outputs[4, 3] = [0.9, 0.95, 0.5, 0.4, 0.5, 1, 0, 0]
    outputs[4, 4] = [0.6, 0.05, 0.5, 0.4, 0.5, 1, 0, 0]  # almost the same box
    boxes, scores, _ = decode(outputs, GRID, suppression_overlap=0.5)
    assert scores.tolist() == [0.9]


def test_loss_weights_follow_the_yolo_rules():
    targets = encode([[0.30, 0.50, 0.16, 0.25]], [2], GRID)
    weights = loss_weights(targets, box_weight=5.0, empty_cell_weight=0.5)
    assert weights[4, 3].tolist() == [1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
    assert weights[0, 0].tolist() == [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
