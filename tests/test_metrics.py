"""Tests of the scoring functions, using cases small enough to check by hand."""

import numpy as np
import pytest

from packaging_defects.boxes import centre_to_corner
from packaging_defects.metrics import (
    average_precision,
    evaluate_detections,
    match_detections,
    operating_point,
    photograph_verdicts,
    precision_recall_curve,
    trapezoid_rule,
)

LEFT = np.array([[0.25, 0.5, 0.2, 0.2]])  # a labelled box on the left, centre format
RIGHT = np.array([[0.75, 0.5, 0.2, 0.2]])
NOTHING = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))


def detection(boxes, scores, classes=None):
    """Build one photograph's detections from centre-format boxes."""
    classes = np.zeros(len(scores), dtype=int) if classes is None else np.array(classes)
    return centre_to_corner(boxes), np.array(scores, dtype=float), classes


def test_trapezoid_rule_is_exact_for_a_straight_line():
    positions = np.array([0.0, 0.5, 2.0])
    assert trapezoid_rule(3.0 * positions + 1.0, positions) == pytest.approx(8.0)


def test_trapezoid_rule_error_falls_with_the_square_of_the_spacing():
    errors = []
    for points in (11, 21):
        positions = np.linspace(0.0, 1.0, points)
        errors.append(abs(trapezoid_rule(positions**2, positions) - 1.0 / 3.0))
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.01)


def test_perfect_detections_score_one():
    result = evaluate_detections(
        [detection(LEFT, [0.9]), detection(RIGHT, [0.8])], [LEFT, RIGHT], [[0], [0]], ["a"]
    )
    assert result["any_defect"]["average_precision"] == pytest.approx(1.0)
    assert result["any_defect"]["precision"] == 1.0 and result["any_defect"]["recall"] == 1.0


def test_no_detections_score_zero():
    result = evaluate_detections([NOTHING], [LEFT], [[0]], ["a"])
    assert result["any_defect"]["average_precision"] == 0.0
    assert result["any_defect"]["recall"] == 0.0


def test_detection_in_the_wrong_place_is_incorrect():
    scores, correct, count = match_detections([detection(RIGHT, [0.9])], [LEFT], [[0]])
    assert correct.tolist() == [False] and count == 1


def test_second_detection_of_the_same_defect_is_incorrect():
    twice = detection(np.vstack([LEFT, LEFT]), [0.9, 0.8])
    scores, correct, count = match_detections([twice], [LEFT], [[0]])
    assert correct.tolist() == [True, False]


def test_overlap_threshold_decides_correctness():
    shifted = LEFT + [0.05, 0.0, 0.0, 0.0]  # overlap is 0.15 / 0.25 = 0.6
    assert match_detections([detection(shifted, [0.9])], [LEFT], [[0]], 0.5)[1].tolist() == [True]
    assert match_detections([detection(shifted, [0.9])], [LEFT], [[0]], 0.75)[1].tolist() == [False]


def test_average_precision_worked_example():
    # Two defects. Detections in confidence order: correct, wrong, correct.
    # Recall goes 0.5, 0.5, 1.0 and precision 1, 1/2, 2/3.  After smoothing,
    # precision is 1 up to recall 0.5 and 2/3 from there to recall 1, so the
    # area is close to 0.5 * 1 + 0.5 * 2/3 = 0.833.
    recall, precision, _ = precision_recall_curve(
        np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2
    )
    assert recall.tolist() == [0.5, 0.5, 1.0]
    assert np.allclose(precision, [1.0, 0.5, 2.0 / 3.0])
    assert average_precision(recall, precision) == pytest.approx(0.8333, abs=0.005)


def test_classes_are_scored_separately():
    # The box is in the right place but given the wrong class.
    wrong_class = [detection(LEFT, [0.9], classes=[1])]
    result = evaluate_detections(wrong_class, [LEFT], [[0]], ["a", "b"])
    assert result["any_defect"]["average_precision"] == pytest.approx(1.0)
    assert result["per_class"]["a"]["average_precision"] == 0.0
    # Class "b" has no labelled defects, so it is left out of the mean.
    assert result["per_class"]["b"]["labelled_defects"] == 0
    assert result["mean_average_precision"] == 0.0


def test_operating_point_chooses_or_applies_a_threshold():
    recall, precision, thresholds = precision_recall_curve(
        np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2
    )
    chosen = operating_point(recall, precision, thresholds)
    assert chosen["threshold"] == pytest.approx(0.7) and chosen["f1"] == pytest.approx(0.8)
    fixed = operating_point(recall, precision, thresholds, confidence_threshold=0.85)
    assert fixed["precision"] == 1.0 and fixed["recall"] == 0.5
    nothing_kept = operating_point(recall, precision, thresholds, confidence_threshold=0.99)
    assert nothing_kept["recall"] == 0.0 and nothing_kept["precision"] == 0.0


def test_photograph_verdicts_count_each_outcome():
    detections = [
        detection(LEFT, [0.9]),  # defective, flagged
        detection(LEFT, [0.2]),  # defective, missed (below the threshold)
        detection(LEFT, [0.8]),  # clean, flagged
        NOTHING,  # clean, passed
        NOTHING,  # clean, passed
    ]
    truth = [LEFT, LEFT, np.zeros((0, 4)), np.zeros((0, 4)), np.zeros((0, 4))]
    verdicts = photograph_verdicts(detections, truth, confidence_threshold=0.5)
    assert verdicts["defective_found"] == 1 and verdicts["defective_missed"] == 1
    assert verdicts["clean_flagged"] == 1 and verdicts["clean_passed"] == 2
    assert verdicts["sensitivity"] == pytest.approx(0.5)
    assert verdicts["specificity"] == pytest.approx(2.0 / 3.0)
    assert verdicts["balanced_accuracy"] == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)
