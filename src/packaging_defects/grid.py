"""Describing defects as regression targets on a grid, the YOLO idea.

A regression network outputs a fixed list of numbers, but a photograph can
contain any number of defects.  YOLO ("You Only Look Once") resolves this by
laying a grid over the photograph and making **every grid cell answer the same
fixed questions**:

| Output | Meaning |
| --- | --- |
| 0 | *Objectness*: is this cell in the middle of a defect? (1 for yes, 0 for no) |
| 1, 2 | Where the centre of the defect lies, measured in cell widths and heights from the top-left corner of this cell |
| 3, 4 | Square roots of the box width and height, as fractions of the image. The square root makes an error of a few pixels matter more for a small box than for a large one. |
| 5 onwards | One number per defect class: 1 for the correct class, else 0 |

Which cells count as "in the middle of a defect"?  Always the cell that
contains the centre of the box.  For a defect that spans several cells, every
cell whose own middle lies in the central part of the box counts as well,
because those cells look just like the centre cell and it would be
contradictory to tell the network that one is a defect and its identical
neighbour is not.  Each of them reports the same box, and the repeats are
merged afterwards.

:func:`encode` turns labelled boxes into this table of targets, and
:func:`decode` turns the network's predicted table back into boxes.  Outputs 1
onwards only mean something in cells that contain a defect, which is why
:func:`loss_weights` switches them off everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boxes import centre_to_corner, non_maximum_suppression


@dataclass(frozen=True)
class Grid:
    """Size of the grid laid over each photograph.

    Attributes
    ----------
    rows, columns : int
        Number of cells down and across.
    number_of_classes : int
        Number of defect classes.
    """

    rows: int = 9
    columns: int = 12
    number_of_classes: int = 5

    @property
    def outputs_per_cell(self) -> int:
        """How many numbers each cell predicts."""
        return 5 + self.number_of_classes


def encode(
    boxes: np.ndarray, classes: np.ndarray, grid: Grid, central_fraction: float = 0.5
) -> np.ndarray:
    """Turn the labelled defects of one photograph into grid targets.

    Parameters
    ----------
    boxes : numpy.ndarray, shape (defects, 4)
        Centre format, fractional coordinates.
    classes : numpy.ndarray, shape (defects,)
        Class number of each defect.
    grid : Grid
    central_fraction : float
        The part of each box, measured from its centre, in which cells are
        asked to report the defect.  0.5 means the central half of the width
        and height.  0 means only the single cell containing the centre.

    Returns
    -------
    numpy.ndarray, shape (rows, columns, outputs per cell)
        The table described in the module documentation.  Cells that report
        no defect are all zero.  Where two defects claim the same cell, the
        smaller defect keeps it, since it has fewer cells to spare.
    """
    targets = np.zeros((grid.rows, grid.columns, grid.outputs_per_cell), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    classes = np.asarray(classes, dtype=int).ravel()
    # Middle of every cell, as a fraction of the image.
    cell_middle_x = (np.arange(grid.columns) + 0.5) / grid.columns
    cell_middle_y = (np.arange(grid.rows) + 0.5) / grid.rows

    # Largest boxes first, so that smaller boxes are written last and win.
    for index in np.argsort(-(boxes[:, 2] * boxes[:, 3]), kind="stable"):
        centre_x, centre_y, width, height = boxes[index]
        # Position of the box centre measured in cells, for example 3.25 = a
        # quarter of the way across the fourth cell.
        across = min(centre_x * grid.columns, grid.columns - 1e-6)
        down = min(centre_y * grid.rows, grid.rows - 1e-6)

        in_central_part = (
            np.abs(cell_middle_y - centre_y)[:, None] <= central_fraction * height / 2
        ) & (np.abs(cell_middle_x - centre_x)[None, :] <= central_fraction * width / 2)
        in_central_part[int(down), int(across)] = True  # the centre cell always reports
        rows, columns = np.nonzero(in_central_part)

        targets[rows, columns] = 0.0
        targets[rows, columns, 0] = 1.0
        targets[rows, columns, 1] = across - columns
        targets[rows, columns, 2] = down - rows
        targets[rows, columns, 3] = np.sqrt(width)
        targets[rows, columns, 4] = np.sqrt(height)
        targets[rows, columns, 5 + classes[index]] = 1.0
    return targets


def loss_weights(
    targets: np.ndarray, box_weight: float = 5.0, empty_cell_weight: float = 0.5
) -> np.ndarray:
    """Decide how much each output of each cell counts in the loss.

    These are the weights $s_{ik}$ in
    :meth:`packaging_defects.network.MultilayerPerceptron.loss_and_gradients`,
    and follow the original YOLO loss:

    * Objectness counts in every cell, but less (``empty_cell_weight``) in the
      many empty cells so they do not drown out the few cells with a defect.
    * Box position and size count only in cells with a defect, and more
      (``box_weight``) because accurate boxes are the point of the exercise.
    * Class outputs count only in cells with a defect.

    Parameters
    ----------
    targets : numpy.ndarray, shape (..., outputs per cell)
        As produced by :func:`encode`.
    box_weight : float
    empty_cell_weight : float

    Returns
    -------
    numpy.ndarray
        Same shape as ``targets``.
    """
    has_defect = targets[..., :1] > 0.5
    weights = np.zeros_like(targets)
    weights[..., :1] = np.where(has_defect, 1.0, empty_cell_weight)
    weights[..., 1:5] = np.where(has_defect, box_weight, 0.0)
    weights[..., 5:] = np.where(has_defect, 1.0, 0.0)
    return weights


def decode(
    outputs: np.ndarray,
    grid: Grid,
    minimum_confidence: float = 0.01,
    suppression_overlap: float = 0.5,
    maximum_detections: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn the predicted table for one photograph back into boxes.

    Parameters
    ----------
    outputs : numpy.ndarray, shape (rows, columns, outputs per cell)
        The network's prediction for every cell.
    grid : Grid
    minimum_confidence : float
        Cells whose objectness is below this are ignored.
    suppression_overlap : float
        Passed to :func:`packaging_defects.boxes.non_maximum_suppression`.
    maximum_detections : int
        At most this many boxes are returned.

    Returns
    -------
    boxes : numpy.ndarray, shape (detections, 4)
        Corner format, fractional coordinates, clipped to the image.
    scores : numpy.ndarray, shape (detections,)
        Confidence of each box, between 0 and 1.
    classes : numpy.ndarray of int, shape (detections,)
        Most likely class of each box.
    """
    outputs = np.asarray(outputs, dtype=float)
    rows, columns = np.meshgrid(np.arange(grid.rows), np.arange(grid.columns), indexing="ij")

    # A regression output is not limited to 0..1, so clip it into that range.
    confidence = np.clip(outputs[..., 0], 0.0, 1.0)
    centre_x = (columns + outputs[..., 1]) / grid.columns
    centre_y = (rows + outputs[..., 2]) / grid.rows
    # Undo the square root; clipping first stops a negative prediction from
    # turning into a positive size when squared.
    width = np.clip(outputs[..., 3], 0.0, 1.0) ** 2
    height = np.clip(outputs[..., 4], 0.0, 1.0) ** 2

    keep = (confidence >= minimum_confidence) & (width > 0) & (height > 0)
    boxes = centre_to_corner(
        np.stack([centre_x[keep], centre_y[keep], width[keep], height[keep]], axis=-1)
    )
    boxes = np.clip(boxes, 0.0, 1.0)
    scores = confidence[keep]
    classes = np.argmax(outputs[..., 5:], axis=-1)[keep]

    kept = non_maximum_suppression(boxes, scores, suppression_overlap)[:maximum_detections]
    return boxes[kept], scores[kept], classes[kept].astype(int)
