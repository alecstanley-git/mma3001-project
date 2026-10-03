"""The main solution: a hand-written detector that follows the YOLO process.

The Ultralytics library, given a folder of labelled photographs, does this:

1. reads the photographs and their label files;
2. lays a grid over each photograph and asks every cell to predict whether a
   defect is centred there, where its box is, and what class it is;
3. trains a neural network to make those predictions by minimising a loss;
4. turns the predictions back into boxes and removes repeats;
5. scores the boxes against the labels.

:class:`GridDetector` does the same five things using only the methods taught
in the unit.  The network is the regression multilayer perceptron of Week 5,
written by hand in :mod:`packaging_defects.network`.

What the network sees
---------------------
One network is shared by every grid cell.  For each cell it is given:

* the block features (see :mod:`packaging_defects.features`) of the blocks in
  and immediately around the cell, in full detail;
* a coarser summary of the wider neighbourhood, so it can judge the size of a
  defect that extends beyond the cell;
* a very coarse summary of the whole photograph, for overall context;
* the position of the cell in the photograph.

Sharing one network between all cells means a defect seen at one place in
training helps to recognise the same defect anywhere else.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .boxes import mirror_boxes
from .dataset import Split
from .features import FEATURES_PER_BLOCK, feature_maps, mirror_feature_maps
from .grid import Grid, decode, encode, loss_weights
from .metrics import average_precision, match_detections, precision_recall_curve
from .network import MultilayerPerceptron, StandardScaler, train

#: Blocks along each side of a grid cell (a cell is 3 by 3 blocks, which is
#: 60 by 60 pixels of the original photograph).
BLOCKS_PER_CELL = 3

#: Cells along each side of a region in the whole-photograph summary.
CELLS_PER_REGION = 3


@dataclass(frozen=True)
class Settings:
    """Every adjustable choice in the detector, with its default value.

    Attributes
    ----------
    hidden_layers : tuple of int
        Neurons in each hidden layer.  An empty tuple gives linear regression.
    detail_reach : int
        How many blocks beyond the centre block the detailed view extends in
        each direction.  3 gives a 7 by 7 block view.
    neighbourhood_reach : int
        How many cells beyond this cell the coarse view extends in each
        direction.  3 gives a 7 by 7 cell view.
    epochs : int
        Maximum passes through the training data.
    batch_size : int
        Grid cells used for each update of the weights.
    learning_rate : float
        Step size of the Adam optimiser.
    penalty : float
        Strength of the penalty on large weights (``alpha`` in the unit notes).
    box_weight, empty_cell_weight : float
        See :func:`packaging_defects.grid.loss_weights`.
    central_fraction : float
        See :func:`packaging_defects.grid.encode`.
    mirrored_copies : bool
        Also train on left-right and top-bottom mirror images of every
        training photograph, which quadruples the training data.
    validation_fraction : float
        Share of training frames set aside to decide when to stop training.
    patience : int
        Passes to wait for improvement before stopping.
    suppression_overlap : float
        Overlap above which two detections are treated as the same defect.
    seed : int
        Seed for everything random, so a run can be repeated exactly.
    """

    hidden_layers: tuple[int, ...] = (128, 64)
    detail_reach: int = 3
    neighbourhood_reach: int = 3
    epochs: int = 60
    batch_size: int = 512
    learning_rate: float = 0.0003
    penalty: float = 0.05
    box_weight: float = 5.0
    empty_cell_weight: float = 0.5
    central_fraction: float = 0.5
    mirrored_copies: bool = True
    validation_fraction: float = 0.15
    patience: int = 10
    suppression_overlap: float = 0.5
    seed: int = 0


class GridDetector:
    """Finds packaging defects by regressing a grid of predictions.

    Parameters
    ----------
    class_names : list of str
        Names of the defect classes, in label-file order.
    settings : Settings, optional
        Adjustable choices; defaults are used if omitted.

    Attributes
    ----------
    network : MultilayerPerceptron or None
        The trained network; ``None`` until :meth:`fit` has run.
    scaler : StandardScaler
        Scaling learned from the training photographs.
    history : dict
        Training loss and validation score for each pass.
    """

    def __init__(self, class_names: list[str], settings: Settings | None = None) -> None:
        self.class_names = list(class_names)
        self.settings = settings or Settings()
        self.grid = Grid(number_of_classes=len(self.class_names))
        self.scaler = StandardScaler()
        self.network: MultilayerPerceptron | None = None
        self.history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Building the network's inputs
    # ------------------------------------------------------------------
    def build_views(self, maps: np.ndarray) -> dict[str, np.ndarray]:
        """Scale block features and build the three views used by every cell.

        Parameters
        ----------
        maps : numpy.ndarray, shape (n, block rows, block columns, 14)

        Returns
        -------
        dict
            ``"detail"`` and ``"neighbourhood"`` are padded with zeros round
            the edge so that cells at the border of the photograph can still
            look "outside" it; after scaling, zero is the average value.
            ``"whole"`` holds one row per photograph.
        """
        count, block_rows, block_columns, features = maps.shape
        expected = (self.grid.rows * BLOCKS_PER_CELL, self.grid.columns * BLOCKS_PER_CELL)
        if (block_rows, block_columns) != expected or features != FEATURES_PER_BLOCK:
            raise ValueError(f"feature maps have shape {maps.shape[1:]}, expected {expected}")
        # Scale a few hundred photographs at a time: scaling everything in one
        # step briefly needs several times the memory of the result.
        scaled = np.empty(maps.shape, dtype=np.float32)
        for start, stop in _chunks(count, 256):
            scaled[start:stop] = self.scaler.transform(maps[start:stop])

        # Average the 3 by 3 blocks of each cell to get one summary per cell.
        cells = scaled.reshape(
            count, self.grid.rows, BLOCKS_PER_CELL, self.grid.columns, BLOCKS_PER_CELL, features
        ).mean(axis=(2, 4))
        # Average again over regions of 3 by 3 cells for the whole-photograph view.
        regions = cells.reshape(
            count,
            self.grid.rows // CELLS_PER_REGION,
            CELLS_PER_REGION,
            self.grid.columns // CELLS_PER_REGION,
            CELLS_PER_REGION,
            features,
        ).mean(axis=(2, 4))

        detail, neighbourhood = self.settings.detail_reach, self.settings.neighbourhood_reach
        return {
            "detail": np.pad(scaled, ((0, 0), (detail, detail), (detail, detail), (0, 0))),
            "neighbourhood": np.pad(
                cells, ((0, 0), (neighbourhood,) * 2, (neighbourhood,) * 2, (0, 0))
            ),
            "whole": regions.reshape(count, -1),
        }

    def cell_inputs(
        self, views: dict[str, np.ndarray], images: np.ndarray, rows: np.ndarray, columns: np.ndarray
    ) -> np.ndarray:
        """Assemble the network's input for a batch of grid cells.

        Parameters
        ----------
        views : dict
            From :meth:`build_views`.
        images, rows, columns : numpy.ndarray of int, shape (batch,)
            Which photograph, and which cell of it, each sample refers to.

        Returns
        -------
        numpy.ndarray, shape (batch, input features)
        """
        batch = len(images)
        detail_width = 2 * self.settings.detail_reach + 1
        neighbourhood_width = 2 * self.settings.neighbourhood_reach + 1
        detail_offsets = np.arange(detail_width)
        neighbourhood_offsets = np.arange(neighbourhood_width)

        # In the padded array the window starts at the cell's centre block
        # minus the reach, plus the padding, which cancel to leave this.
        first_block_row = rows * BLOCKS_PER_CELL + 1
        first_block_column = columns * BLOCKS_PER_CELL + 1
        detail = views["detail"][
            images[:, None, None],
            first_block_row[:, None, None] + detail_offsets[None, :, None],
            first_block_column[:, None, None] + detail_offsets[None, None, :],
        ]
        neighbourhood = views["neighbourhood"][
            images[:, None, None],
            rows[:, None, None] + neighbourhood_offsets[None, :, None],
            columns[:, None, None] + neighbourhood_offsets[None, None, :],
        ]
        # Cell position, scaled to run from -1 at one edge to 1 at the other.
        position = np.stack(
            [2.0 * rows / (self.grid.rows - 1) - 1.0, 2.0 * columns / (self.grid.columns - 1) - 1.0],
            axis=1,
        ).astype(np.float32)
        return np.concatenate(
            [
                detail.reshape(batch, -1),
                neighbourhood.reshape(batch, -1),
                views["whole"][images],
                position,
            ],
            axis=1,
        )

    def number_of_inputs(self) -> int:
        """Length of the list of numbers describing one grid cell."""
        detail_width = 2 * self.settings.detail_reach + 1
        neighbourhood_width = 2 * self.settings.neighbourhood_reach + 1
        regions = (self.grid.rows // CELLS_PER_REGION) * (self.grid.columns // CELLS_PER_REGION)
        return FEATURES_PER_BLOCK * (detail_width**2 + neighbourhood_width**2 + regions) + 2

    def all_cells(self, number_of_images: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """List every cell of every photograph as (image, row, column) arrays."""
        images, rows, columns = np.meshgrid(
            np.arange(number_of_images),
            np.arange(self.grid.rows),
            np.arange(self.grid.columns),
            indexing="ij",
        )
        return images.ravel(), rows.ravel(), columns.ravel()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        training: Split,
        cache_file: Path | None = None,
        report: Callable[[str], None] | None = None,
    ) -> dict[str, list[float]]:
        """Train the detector.

        A share of the training frames is set aside to decide when to stop.
        Copies of the same original frame are always kept on the same side of
        that division, otherwise the set-aside photographs would be near
        duplicates of ones used for training and the stopping decision would
        be over-optimistic.

        Parameters
        ----------
        training : Split
            Training photographs and labels.
        cache_file : pathlib.Path, optional
            Where to keep the block features between runs.
        report : callable, optional
            Receives progress messages, for example ``print``.

        Returns
        -------
        dict
            Training loss and validation score for each pass.
        """
        settings = self.settings
        generator = np.random.default_rng(settings.seed)
        maps = feature_maps(training.image_paths, cache_file)

        # Divide by original frame, not by photograph.
        frames = sorted(set(training.sources))
        generator.shuffle(frames)
        held_back = set(frames[: max(1, int(round(settings.validation_fraction * len(frames))))])
        is_held_back = np.array([source in held_back for source in training.sources])
        fitting_images = np.flatnonzero(~is_held_back)
        checking_images = np.flatnonzero(is_held_back)

        # Scaling is learned from the photographs used for fitting only.
        self.scaler.fit(maps[fitting_images])

        # Build the fitting set, adding mirror images if requested.
        mirrors = [(False, False)]
        if settings.mirrored_copies:
            mirrors += [(True, False), (False, True), (True, True)]
        views = self.build_views(
            np.concatenate([mirror_feature_maps(maps[fitting_images], *mirror) for mirror in mirrors])
        )
        targets = np.stack(
            [
                encode(
                    mirror_boxes(training.boxes[i], *mirror),
                    training.classes[i],
                    self.grid,
                    settings.central_fraction,
                )
                for mirror in mirrors
                for i in fitting_images
            ]
        )
        weights = loss_weights(targets, settings.box_weight, settings.empty_cell_weight)
        images, rows, columns = self.all_cells(len(targets))

        def make_batch(sample_numbers: np.ndarray):
            chosen = images[sample_numbers], rows[sample_numbers], columns[sample_numbers]
            return self.cell_inputs(views, *chosen), targets[chosen], weights[chosen]

        checking_views = self.build_views(maps[checking_images])
        checking_boxes = [training.boxes[i] for i in checking_images]
        checking_classes = [training.classes[i] for i in checking_images]

        def validation_score() -> float:
            detections = self.detect_from_views(checking_views)
            scores, correct, count = match_detections(detections, checking_boxes, checking_classes)
            recall, precision, _ = precision_recall_curve(scores, correct, count)
            return average_precision(recall, precision)

        self.network = MultilayerPerceptron(
            (self.number_of_inputs(), *settings.hidden_layers, self.grid.outputs_per_cell),
            seed=settings.seed,
        )
        if report is not None:
            report(
                f"training on {len(fitting_images)} photographs "
                f"({len(images)} grid cells including mirror images), "
                f"stopping rule uses {len(checking_images)}; "
                f"network has {self.network.number_of_parameters()} parameters"
            )
        self.history = train(
            self.network,
            make_batch,
            len(images),
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            penalty=settings.penalty,
            validation_score=validation_score,
            patience=settings.patience,
            seed=settings.seed,
            report=report,
        )
        return self.history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def detect_from_views(self, views: dict[str, np.ndarray]) -> list[tuple]:
        """Run the network on every cell of prepared photographs and decode."""
        count = len(views["whole"])
        images, rows, columns = self.all_cells(count)
        outputs = np.concatenate(
            [
                self.network.predict(
                    self.cell_inputs(
                        views, images[start:stop], rows[start:stop], columns[start:stop]
                    )
                )
                for start, stop in _chunks(len(images), 8192)
            ]
        ).reshape(count, self.grid.rows, self.grid.columns, self.grid.outputs_per_cell)
        return [
            decode(output, self.grid, suppression_overlap=self.settings.suppression_overlap)
            for output in outputs
        ]

    def detect(self, image_paths: list[Path], cache_file: Path | None = None) -> list[tuple]:
        """Find defects in photographs.

        Parameters
        ----------
        image_paths : list of pathlib.Path
            Photographs to examine.
        cache_file : pathlib.Path, optional
            Where to keep the block features between runs.

        Returns
        -------
        list of (boxes, scores, classes)
            One entry per photograph: boxes in corner format with fractional
            coordinates, a confidence between 0 and 1 per box, and a class
            number per box.

        Raises
        ------
        RuntimeError
            If the detector has not been trained or loaded.
        OSError
            If a photograph cannot be opened.
        """
        if self.network is None:
            raise RuntimeError("the detector must be trained or loaded before use")
        if len(image_paths) == 0:
            return []
        return self.detect_from_views(self.build_views(feature_maps(image_paths, cache_file)))

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Write the trained detector to a file."""
        if self.network is None:
            raise RuntimeError("there is no trained network to save")
        state = {
            "class_names": self.class_names,
            "settings": asdict(self.settings),
            "scaler_mean": self.scaler.mean,
            "scaler_scale": self.scaler.scale,
            "layer_sizes": self.network.layer_sizes,
            "parameters": self.network.copy_parameters(),
            "history": self.history,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as handle:
            pickle.dump(state, handle)

    @classmethod
    def load(cls, path: Path) -> "GridDetector":
        """Read a detector written by :meth:`save`.

        Only open files you created yourself: this uses Python's ``pickle``,
        which can run code hidden in a file from an untrusted source.
        """
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        settings = dict(state["settings"])
        settings["hidden_layers"] = tuple(settings["hidden_layers"])
        detector = cls(state["class_names"], Settings(**settings))
        detector.scaler.mean, detector.scaler.scale = state["scaler_mean"], state["scaler_scale"]
        detector.network = MultilayerPerceptron(state["layer_sizes"])
        detector.network.set_parameters(state["parameters"])
        detector.history = state["history"]
        return detector


def _chunks(total: int, size: int):
    """Yield (start, stop) pairs that cover ``range(total)`` in pieces."""
    for start in range(0, total, size):
        yield start, min(start + size, total)
