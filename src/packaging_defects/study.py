"""Measuring how each design choice affects the detector.

The unit notes (section 5.4) recommend investigating architecture,
regularisation and random seed rather than trusting a single run.  This module
retrains the detector with one setting changed at a time and scores every
version on the **validation** photographs.  The test photographs are not
touched, so they remain an honest final check.

The variations are:

* **Network size**, from no hidden layer at all (which is ordinary linear
  regression, the baseline a flexible model has to beat) to a larger network.
* **Weight penalty**, weaker and stronger.
* **Mirrored copies** of the training photographs, switched off.
* **Central fraction**, set to zero so only one cell reports each defect, as
  in the original YOLO.
* **Random seed**, to show how much the score moves by chance alone.  A
  difference between two settings that is smaller than this spread should not
  be read as real.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from .dataset import load_dataset, read_class_names
from .detector import GridDetector, Settings
from .features import feature_maps
from .metrics import evaluate_detections


def variations(chosen: Settings) -> list[tuple[str, Settings]]:
    """List the settings to compare, each with a plain-language label.

    Parameters
    ----------
    chosen : Settings
        The settings used for the final detector.

    Returns
    -------
    list of (label, Settings)
    """
    return [
        ("chosen settings (seed 0)", chosen),
        ("seed 1", replace(chosen, seed=1)),
        ("seed 2", replace(chosen, seed=2)),
        ("seed 3", replace(chosen, seed=3)),
        ("no hidden layer (linear regression)", replace(chosen, hidden_layers=())),
        ("one hidden layer of 64", replace(chosen, hidden_layers=(64,))),
        ("hidden layers of 256 and 128", replace(chosen, hidden_layers=(256, 128))),
        ("weight penalty 10 times weaker", replace(chosen, penalty=chosen.penalty / 10)),
        ("weight penalty 10 times stronger", replace(chosen, penalty=chosen.penalty * 10)),
        ("no mirrored copies", replace(chosen, mirrored_copies=False)),
        ("only the centre cell reports a defect", replace(chosen, central_fraction=0.0)),
    ]


def run_one(job: tuple[str, Settings, Path, Path]) -> dict:
    """Train one variation and score it on validation.

    Parameters
    ----------
    job : tuple
        ``(label, settings, dataset_directory, cache_directory)``.  Packed
        into one argument so it can be handed to a separate process.

    Returns
    -------
    dict
        The label, headline scores, network size and training time.
    """
    label, settings, dataset_directory, cache_directory = job
    class_names = read_class_names(dataset_directory)
    data = load_dataset(dataset_directory)
    detector = GridDetector(class_names, settings)
    start = time.perf_counter()
    history = detector.fit(data["train"], cache_directory / "train.npz")
    training_seconds = time.perf_counter() - start

    valid = data["valid"]
    detections = detector.detect(valid.image_paths, cache_directory / "valid.npz")
    result = evaluate_detections(detections, valid.boxes, valid.classes, class_names)
    summary = {
        "label": label,
        "average_precision": result["any_defect"]["average_precision"],
        "mean_average_precision": result["mean_average_precision"],
        "parameters": detector.network.number_of_parameters(),
        "passes": len(history["loss"]),
        "training_seconds": training_seconds,
    }
    print(
        f"{label:40s} average precision {summary['average_precision']:.3f}  "
        f"parameters {summary['parameters']:>7,}  passes {summary['passes']:2d}  "
        f"{training_seconds:5.0f} s",
        flush=True,
    )
    return summary


def run_study(dataset_directory: Path, cache_directory: Path, workers: int = 2) -> list[dict]:
    """Train and score every variation.

    Parameters
    ----------
    dataset_directory : pathlib.Path
        Folder containing ``data.yaml`` and the split folders.
    cache_directory : pathlib.Path
        Folder for saved block features.
    workers : int
        Trainings to run at the same time.  Each needs roughly 4 gigabytes
        of memory.

    Returns
    -------
    list of dict
        One :func:`run_one` summary per variation, in the order of
        :func:`variations`.
    """
    # Compute the block features once here so the separate processes all read
    # the saved copy instead of racing to write it.
    data = load_dataset(dataset_directory)
    feature_maps(data["train"].image_paths, cache_directory / "train.npz")
    feature_maps(data["valid"].image_paths, cache_directory / "valid.npz")

    jobs = [
        (label, settings, Path(dataset_directory), Path(cache_directory))
        for label, settings in variations(Settings())
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, jobs))
