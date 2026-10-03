"""Geometry of bounding boxes.

A bounding box is a rectangle drawn around one defect.  Two ways of writing a
box down are used in this project:

* **Centre format** ``(centre x, centre y, width, height)`` is how the YOLO
  label files store boxes and how the network predicts them.
* **Corner format** ``(left, top, right, bottom)`` is easier for measuring how
  much two boxes overlap.

Every coordinate is a *fraction of the image size*, so ``0.5`` means "half way
across" whatever the resolution.  This keeps the code independent of the image
resolution, and the overlap measure below is unchanged by the choice of units.
"""

from __future__ import annotations

import numpy as np


def centre_to_corner(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from centre format to corner format.

    Parameters
    ----------
    boxes : numpy.ndarray, shape (n, 4)
        Rows of ``(centre x, centre y, width, height)``.

    Returns
    -------
    numpy.ndarray, shape (n, 4)
        Rows of ``(left, top, right, bottom)``.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    half_size = boxes[:, 2:] / 2.0
    return np.hstack([boxes[:, :2] - half_size, boxes[:, :2] + half_size])


def corner_to_centre(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from corner format to centre format.

    Parameters
    ----------
    boxes : numpy.ndarray, shape (n, 4)
        Rows of ``(left, top, right, bottom)``.

    Returns
    -------
    numpy.ndarray, shape (n, 4)
        Rows of ``(centre x, centre y, width, height)``.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    size = boxes[:, 2:] - boxes[:, :2]
    return np.hstack([boxes[:, :2] + size / 2.0, size])


def mirror_boxes(boxes: np.ndarray, left_right: bool, up_down: bool) -> np.ndarray:
    """Move boxes to where they would be in a mirrored photograph.

    Parameters
    ----------
    boxes : numpy.ndarray, shape (n, 4)
        Centre format, fractional coordinates.
    left_right : bool
        Mirror left to right.
    up_down : bool
        Mirror top to bottom.

    Returns
    -------
    numpy.ndarray, shape (n, 4)
        Centre format.  Width and height are unchanged.
    """
    boxes = np.array(boxes, dtype=float).reshape(-1, 4)
    if left_right:
        boxes[:, 0] = 1.0 - boxes[:, 0]
    if up_down:
        boxes[:, 1] = 1.0 - boxes[:, 1]
    return boxes


def intersection_over_union(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Measure the overlap between every pair of boxes from two sets.

    The measure is the area shared by two boxes divided by the total area they
    cover together.  It is 1 for identical boxes and 0 for boxes that do not
    touch.  It is the standard test of whether a predicted box is "close
    enough" to a labelled one.

    Parameters
    ----------
    first : numpy.ndarray, shape (n, 4)
        Boxes in corner format.
    second : numpy.ndarray, shape (m, 4)
        Boxes in corner format.

    Returns
    -------
    numpy.ndarray, shape (n, m)
        Entry ``[i, j]`` is the overlap between ``first[i]`` and ``second[j]``.
    """
    first = np.asarray(first, dtype=float).reshape(-1, 4)
    second = np.asarray(second, dtype=float).reshape(-1, 4)

    # Broadcasting compares each box in `first` with each box in `second`.
    shared_left_top = np.maximum(first[:, None, :2], second[None, :, :2])
    shared_right_bottom = np.minimum(first[:, None, 2:], second[None, :, 2:])
    # Boxes that do not touch would give a negative width or height.
    shared_size = np.clip(shared_right_bottom - shared_left_top, 0.0, None)
    shared_area = shared_size[..., 0] * shared_size[..., 1]

    first_area = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    second_area = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
    combined_area = first_area[:, None] + second_area[None, :] - shared_area

    # Guard against dividing by zero for boxes with no area.
    return shared_area / np.maximum(combined_area, 1e-12)


def non_maximum_suppression(
    boxes: np.ndarray, scores: np.ndarray, overlap_threshold: float = 0.5
) -> np.ndarray:
    """Remove repeated detections of the same defect.

    Neighbouring grid cells often report the same defect.  This keeps the most
    confident box, discards every other box that overlaps it by more than
    ``overlap_threshold``, and repeats with the boxes that remain.

    Parameters
    ----------
    boxes : numpy.ndarray, shape (n, 4)
        Boxes in corner format.
    scores : numpy.ndarray, shape (n,)
        Confidence of each box; higher means more confident.
    overlap_threshold : float
        Boxes overlapping a kept box by more than this are discarded.

    Returns
    -------
    numpy.ndarray of int
        Indices of the boxes to keep, most confident first.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).ravel()
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError("boxes and scores must have the same length")

    remaining = np.argsort(-scores, kind="stable")
    kept = []
    while remaining.size > 0:
        best = remaining[0]
        kept.append(best)
        overlaps = intersection_over_union(boxes[best], boxes[remaining[1:]])[0]
        remaining = remaining[1:][overlaps <= overlap_threshold]
    return np.array(kept, dtype=int)
