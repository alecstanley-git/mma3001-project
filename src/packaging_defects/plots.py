"""Figures for the report.

Line charts are saved as scalable vector graphics (``.svg``), which stay sharp
at any size.  The figure of example photographs is saved as a ``.png`` image.

Every figure is drawn at the width it occupies on the page of the report, so
the text inside the figures comes out the same size as the text around them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; no window is needed
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (importing it makes the "science" style available)
from PIL import Image

#: Width of the text on a page of the report, in inches (A4 paper with the
#: default margins of the Typst typesetting program).
REPORT_TEXT_WIDTH = 6.30

#: Width of an ordinary figure, which the report shows at 70% of the text width.
FIG_WIDTH = 0.70 * REPORT_TEXT_WIDTH

# The "science" style typesets labels with LaTeX.  On a computer without
# LaTeX, the same style is used with ordinary fonts so the commands still run.
plt.style.use("science" if shutil.which("latex") else ["science", "no-latex"])
plt.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_WIDTH * 2.625 / 3.5)


def _save(figure, path: Path) -> None:
    """Write a figure to disk, creating the folder if needed, and close it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=250)
    plt.close(figure)


def plot_training_history(history: dict[str, list[float]], path: Path) -> None:
    """Plot the training loss and the validation score for each pass.

    The left panel should fall steadily.  The right panel shows the score on
    frames held back from training; the dashed line marks the pass whose
    weights were kept.

    Parameters
    ----------
    history : dict
        As returned by :meth:`packaging_defects.detector.GridDetector.fit`.
    path : pathlib.Path
        File to write.
    """
    passes = np.arange(1, len(history["loss"]) + 1)
    # Two panels side by side, shown across the full width of the page.
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(REPORT_TEXT_WIDTH, 0.36 * REPORT_TEXT_WIDTH)
    )
    left.plot(passes, history["loss"], color="tab:blue")
    left.set(xlabel="Pass through the training data", ylabel="Training loss", yscale="log")
    left.set_title("Training loss")

    scores = history.get("validation_score", [])
    if scores:
        best = int(np.argmax(scores)) + 1
        right.plot(passes, scores, color="tab:orange")
        # The dashed line marks the pass whose weights were kept.
        right.axvline(best, color="black", linestyle="--")
    right.set(xlabel="Pass through the training data", ylabel="Average precision")
    right.set_title(f"Score on held-back frames (best: pass {best})" if scores else "")
    for axes in (left, right):
        axes.grid(True, alpha=0.3)
    _save(figure, path)


def plot_precision_recall(curves: dict[str, dict], path: Path) -> None:
    """Plot precision against recall for one or more detectors.

    Parameters
    ----------
    curves : dict
        Maps a label to the ``"any_defect"`` entry returned by
        :func:`packaging_defects.metrics.evaluate_detections`.
    path : pathlib.Path
        File to write.
    """
    figure, axes = plt.subplots()
    for label, result in curves.items():
        axes.plot(
            result["curve"]["recall"],
            result["curve"]["precision"],
            label=f"{label} (average precision {result['average_precision']:.2f})",
        )
    axes.set(xlabel="Recall (share of defects found)", ylabel="Precision (share of detections correct)")
    axes.set(xlim=(0, 1), ylim=(0, 1.02))
    axes.grid(True, alpha=0.3)
    # The legend sits above the plot so that it cannot cover the curves.
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    _save(figure, path)


def plot_examples(
    image_paths: list[Path],
    detections: list[tuple],
    truth_boxes: list[np.ndarray],
    confidence_threshold: float,
    path: Path,
    columns: int = 3,
) -> None:
    """Draw labelled and detected boxes over photographs.

    Labelled defects are drawn in green and detections in red, with their
    confidence.  Only detections at or above ``confidence_threshold`` are
    drawn.

    Parameters
    ----------
    image_paths : list of pathlib.Path
        Photographs to show.
    detections : list of (boxes, scores, classes)
        One entry per photograph; boxes in corner format.
    truth_boxes : list of numpy.ndarray
        Labelled boxes per photograph, centre format.
    confidence_threshold : float
    path : pathlib.Path
        File to write.
    columns : int
        Photographs per row.
    """
    rows = int(np.ceil(len(image_paths) / columns))
    # Shown across the full width of the page; photographs are 4 wide by 3 high.
    panel_width = REPORT_TEXT_WIDTH / columns
    figure, grid = plt.subplots(
        rows,
        columns,
        figsize=(REPORT_TEXT_WIDTH, 0.78 * panel_width * rows + 0.3),
        squeeze=False,
    )
    for axes in grid.ravel():
        axes.axis("off")
    for axes, image_path, (boxes, scores, _), labelled in zip(
        grid.ravel(), image_paths, detections, truth_boxes
    ):
        with Image.open(image_path) as image:
            width, height = image.size
            axes.imshow(image)
        for centre_x, centre_y, box_width, box_height in np.asarray(labelled).reshape(-1, 4):
            axes.add_patch(
                plt.Rectangle(
                    ((centre_x - box_width / 2) * width, (centre_y - box_height / 2) * height),
                    box_width * width,
                    box_height * height,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )
        for (left, top, right, bottom), score in zip(boxes, scores):
            if score < confidence_threshold:
                continue
            axes.add_patch(
                plt.Rectangle(
                    (left * width, top * height),
                    (right - left) * width,
                    (bottom - top) * height,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                    linestyle="--",
                )
            )
            axes.text(
                left * width,
                top * height - 8,
                f"{score:.2f}",
                color="red",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
            )
    figure.suptitle("Green: labelled defect.  Red dashed: detection with its confidence.")
    _save(figure, path)


def plot_study(results: list[dict], path: Path) -> None:
    """Bar chart of validation average precision for each studied setting.

    Parameters
    ----------
    results : list of dict
        Each with ``"label"`` and ``"average_precision"``, as written by the
        ``study`` command.
    path : pathlib.Path
        File to write.
    """
    labels = [result["label"] for result in results]
    scores = [result["average_precision"] for result in results]
    # Shown across the full width of the page so the long labels fit.
    figure, axes = plt.subplots(figsize=(REPORT_TEXT_WIDTH, 0.22 * len(labels) + 0.7))
    bars = axes.barh(labels[::-1], scores[::-1], color="tab:blue")
    axes.bar_label(bars, fmt="%.3f", padding=3)
    axes.set(xlabel="Average precision on the validation photographs", xlim=(0, max(scores) * 1.18))
    axes.grid(True, axis="x", alpha=0.3)
    _save(figure, path)
