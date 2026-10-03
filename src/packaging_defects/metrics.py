"""Scoring detections against the labelled defects.

The same functions score both the hand-written detector and the Ultralytics
baseline, so the comparison between them is like for like.

How a detection is judged
-------------------------
A detection is **correct** when it overlaps a labelled defect by at least
``overlap_threshold`` (intersection over union, see
:func:`packaging_defects.boxes.intersection_over_union`) and that defect has
not already been claimed by a more confident detection.  From this:

* **Precision** - of the detections reported, the fraction that are correct.
* **Recall** - of the labelled defects, the fraction that were found.
* **Average precision** - the area under the curve of precision against
  recall as the confidence threshold is lowered.  It summarises the detector
  across *all* thresholds in one number between 0 and 1.

Throughout, "detections" means a list with one ``(boxes, scores, classes)``
entry per photograph, with boxes in corner format and fractional coordinates.
"""

from __future__ import annotations

import numpy as np

from .boxes import centre_to_corner, intersection_over_union


def trapezoid_rule(heights: np.ndarray, positions: np.ndarray) -> float:
    """Integrate sampled values with the composite trapezoid rule.

    This is the first Newton-Cotes formula from the unit: the area under each
    pair of neighbouring samples is approximated by a trapezoid.

    Parameters
    ----------
    heights : numpy.ndarray
        Values of the function at each position.
    positions : numpy.ndarray
        Where the function was sampled, in increasing order.

    Returns
    -------
    float
        Approximate area under the curve.
    """
    heights = np.asarray(heights, dtype=float)
    positions = np.asarray(positions, dtype=float)
    widths = np.diff(positions)
    return float(np.sum(widths * (heights[:-1] + heights[1:]) / 2.0))


def match_detections(
    detections: list[tuple],
    truth_boxes: list[np.ndarray],
    truth_classes: list[np.ndarray],
    overlap_threshold: float = 0.5,
    class_number: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Decide which detections are correct.

    Within each photograph, detections are taken from most to least confident.
    Each claims the unclaimed labelled defect it overlaps most, provided the
    overlap reaches ``overlap_threshold``.

    Parameters
    ----------
    detections : list of (boxes, scores, classes)
        One entry per photograph.
    truth_boxes : list of numpy.ndarray
        Labelled boxes per photograph, centre format.
    truth_classes : list of numpy.ndarray
        Labelled class numbers per photograph.
    overlap_threshold : float
        Minimum overlap for a detection to count as correct.
    class_number : int, optional
        Score only this class.  ``None`` ignores classes and asks only
        "was a defect found here?".

    Returns
    -------
    scores : numpy.ndarray
        Confidence of every detection considered, across all photographs.
    correct : numpy.ndarray of bool
        Whether each of those detections was correct.
    number_of_defects : int
        How many labelled defects there were to find.
    """
    all_scores, all_correct, number_of_defects = [], [], 0
    for (boxes, scores, classes), labelled_boxes, labelled_classes in zip(
        detections, truth_boxes, truth_classes
    ):
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).ravel()
        labelled_boxes = np.asarray(labelled_boxes, dtype=float).reshape(-1, 4)
        if class_number is not None:
            keep = np.asarray(classes).ravel() == class_number
            boxes, scores = boxes[keep], scores[keep]
            labelled_boxes = labelled_boxes[np.asarray(labelled_classes).ravel() == class_number]
        number_of_defects += len(labelled_boxes)

        order = np.argsort(-scores, kind="stable")
        correct = np.zeros(len(order), dtype=bool)
        if len(labelled_boxes) and len(order):
            overlaps = intersection_over_union(boxes[order], centre_to_corner(labelled_boxes))
            claimed = np.zeros(len(labelled_boxes), dtype=bool)
            for row in range(len(order)):
                candidates = np.where(~claimed, overlaps[row], -1.0)
                best = int(np.argmax(candidates))
                if candidates[best] >= overlap_threshold:
                    claimed[best] = True
                    correct[row] = True
        all_scores.append(scores[order])
        all_correct.append(correct)

    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    correct = np.concatenate(all_correct) if all_correct else np.zeros(0, dtype=bool)
    return scores, correct, number_of_defects


def precision_recall_curve(
    scores: np.ndarray, correct: np.ndarray, number_of_defects: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace precision and recall as the confidence threshold is lowered.

    Parameters
    ----------
    scores, correct, number_of_defects
        As returned by :func:`match_detections`.

    Returns
    -------
    recall, precision, thresholds : numpy.ndarray
        Entry ``i`` gives the recall and precision obtained when only
        detections with confidence of at least ``thresholds[i]`` are kept.
    """
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(correct[order])
    reported = np.arange(1, len(order) + 1)
    recall = hits / max(number_of_defects, 1)
    precision = hits / reported
    return recall, precision, scores[order]


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve.

    The raw curve zig-zags, so it is first smoothed in the standard way used
    by detection benchmarks, then integrated:

    1. At each recall level, take the best precision achievable at that
       recall *or any higher recall*.  This removes the zig-zags.
    2. Sample that smoothed curve at 101 evenly spaced recall levels from 0
       to 1.  Recall levels the detector never reaches count as precision 0.
    3. Integrate the samples with the trapezoid rule.

    Parameters
    ----------
    recall, precision : numpy.ndarray
        As returned by :func:`precision_recall_curve`.

    Returns
    -------
    float
        Between 0 (useless) and 1 (every defect found with no false alarms).
    """
    if len(recall) == 0:
        return 0.0
    # Running maximum taken from the high-recall end.
    best_from_here_on = np.maximum.accumulate(precision[::-1])[::-1]
    sample_points = np.linspace(0.0, 1.0, 101)
    # For each sample point, the first position on the curve that reaches it.
    # The tiny allowance stops rounding in 0.1 + 0.2 style sums from mattering.
    first_reaching = np.searchsorted(recall, sample_points - 1e-9, side="left")
    reached = first_reaching < len(recall)
    sampled = np.zeros_like(sample_points)
    sampled[reached] = best_from_here_on[first_reaching[reached]]
    return trapezoid_rule(sampled, sample_points)


def operating_point(
    recall: np.ndarray,
    precision: np.ndarray,
    thresholds: np.ndarray,
    confidence_threshold: float | None = None,
) -> dict[str, float]:
    """Report precision and recall at one confidence threshold.

    A detector in use needs a single threshold: detections below it are
    discarded.  If ``confidence_threshold`` is ``None`` the threshold giving
    the highest F1 score is chosen.  The F1 score, ``2PR / (P + R)``, is high
    only when precision and recall are both high.  Choose the threshold this
    way on validation data, then pass that value in when scoring test data.

    Parameters
    ----------
    recall, precision, thresholds : numpy.ndarray
        As returned by :func:`precision_recall_curve`.
    confidence_threshold : float, optional
        A threshold chosen beforehand.

    Returns
    -------
    dict
        ``"threshold"``, ``"precision"``, ``"recall"`` and ``"f1"``.  All
        scores are zero if no detection reaches the threshold.
    """
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    if confidence_threshold is None:
        if len(f1) == 0:
            return {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        chosen = int(np.argmax(f1))
        confidence_threshold = float(thresholds[chosen])
    else:
        # Thresholds are in decreasing order; find the last one still kept.
        kept = int(np.sum(thresholds >= confidence_threshold))
        if kept == 0:
            return {
                "threshold": float(confidence_threshold),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        chosen = kept - 1
    return {
        "threshold": float(confidence_threshold),
        "precision": float(precision[chosen]),
        "recall": float(recall[chosen]),
        "f1": float(f1[chosen]),
    }


def evaluate_detections(
    detections: list[tuple],
    truth_boxes: list[np.ndarray],
    truth_classes: list[np.ndarray],
    class_names: list[str],
    overlap_threshold: float = 0.5,
    confidence_threshold: float | None = None,
) -> dict:
    """Score a set of detections in the ways the report needs.

    Parameters
    ----------
    detections : list of (boxes, scores, classes)
        One entry per photograph.
    truth_boxes, truth_classes : list of numpy.ndarray
        Labelled defects per photograph.
    class_names : list of str
        Names of the defect classes.
    overlap_threshold : float
        Minimum overlap for a detection to count as correct.
    confidence_threshold : float, optional
        Passed to :func:`operating_point`.

    Returns
    -------
    dict
        * ``"any_defect"`` - classes ignored: average precision, precision
          and recall at one threshold, and the curve itself.  This answers the main
          engineering question, "is there a defect, and where?".
        * ``"per_class"`` - average precision and number of labelled defects
          for each class.
        * ``"mean_average_precision"`` - the per-class values averaged over
          the classes that have at least one labelled defect.
    """
    scores, correct, count = match_detections(
        detections, truth_boxes, truth_classes, overlap_threshold
    )
    recall, precision, thresholds = precision_recall_curve(scores, correct, count)
    any_defect = {
        "average_precision": average_precision(recall, precision),
        "labelled_defects": count,
        **operating_point(recall, precision, thresholds, confidence_threshold),
        "curve": {"recall": recall.tolist(), "precision": precision.tolist()},
    }

    per_class = {}
    for class_number, class_name in enumerate(class_names):
        class_scores, class_correct, class_count = match_detections(
            detections, truth_boxes, truth_classes, overlap_threshold, class_number
        )
        class_recall, class_precision, _ = precision_recall_curve(
            class_scores, class_correct, class_count
        )
        per_class[class_name] = {
            "average_precision": average_precision(class_recall, class_precision),
            "labelled_defects": class_count,
        }
    present = [entry["average_precision"] for entry in per_class.values() if entry["labelled_defects"]]
    return {
        "overlap_threshold": overlap_threshold,
        "any_defect": any_defect,
        "per_class": per_class,
        "mean_average_precision": float(np.mean(present)) if present else 0.0,
    }


def photograph_verdicts(
    detections: list[tuple], truth_boxes: list[np.ndarray], confidence_threshold: float
) -> dict[str, float]:
    """Score the simpler question "is this pack defective or not?".

    A photograph is flagged as defective when any detection reaches
    ``confidence_threshold``.  This is how a packing line would use the
    detector: flag a pack for a person to inspect.

    Parameters
    ----------
    detections : list of (boxes, scores, classes)
    truth_boxes : list of numpy.ndarray
        A photograph is truly defective if it has at least one labelled box.
    confidence_threshold : float
        Chosen beforehand on validation data, never on the data being scored.

    Returns
    -------
    dict
        * ``"sensitivity"`` - fraction of defective photographs flagged.
        * ``"specificity"`` - fraction of clean photographs left unflagged.
        * ``"balanced_accuracy"`` - the mean of those two.  Unlike plain
          accuracy, a detector cannot score well here by flagging everything.
        * the four counts behind them.
    """
    flagged = np.array(
        [bool(np.any(np.asarray(scores) >= confidence_threshold)) for _, scores, _ in detections]
    )
    defective = np.array([len(boxes) > 0 for boxes in truth_boxes])
    found = int(np.sum(flagged & defective))
    missed = int(np.sum(~flagged & defective))
    false_alarms = int(np.sum(flagged & ~defective))
    correctly_passed = int(np.sum(~flagged & ~defective))
    sensitivity = found / max(found + missed, 1)
    specificity = correctly_passed / max(correctly_passed + false_alarms, 1)
    return {
        "confidence_threshold": float(confidence_threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "defective_found": found,
        "defective_missed": missed,
        "clean_flagged": false_alarms,
        "clean_passed": correctly_passed,
    }
