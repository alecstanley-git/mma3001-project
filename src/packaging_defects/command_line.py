"""Command-line entry point: ``packaging-defects <command>``.

| Command | What it does |
| --- | --- |
| `train` | Train the hand-written detector and score it on validation. |
| `evaluate` | Score the trained detector once on the test photographs. |
| `baseline` | Train and score the Ultralytics alternative the same way. |
| `compare` | Put the two sets of test results side by side. |
| `study` | Measure how each design choice changes the validation score. |
| `profile` | Time the detector and find where the time goes. |

The roles of the three parts of the dataset follow the unit notes:
**training** photographs fit the network, **validation** photographs choose
settings and the confidence threshold, and **test** photographs are used once,
at the end, for the reported result.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

from . import plots
from .dataset import Split, load_dataset, read_class_names
from .detector import GridDetector, Settings
from .metrics import evaluate_detections, photograph_verdicts

DETECTOR_FILE = "detector.pkl"


def score(
    detections: list[tuple],
    split: Split,
    class_names: list[str],
    confidence_threshold: float | None = None,
) -> dict:
    """Score detections on one part of the dataset.

    Parameters
    ----------
    detections : list of (boxes, scores, classes)
        One entry per photograph of ``split``.
    split : Split
        The photographs and their labels.
    class_names : list of str
    confidence_threshold : float, optional
        Threshold chosen beforehand on validation data.  ``None`` chooses the
        best threshold on this data, which is only appropriate for validation.

    Returns
    -------
    dict
        Detection scores at the usual overlap of 0.5, average precision at
        the stricter overlap of 0.75, and the defective-or-clean verdicts.
    """
    detection = evaluate_detections(
        detections, split.boxes, split.classes, class_names, 0.5, confidence_threshold
    )
    strict = evaluate_detections(detections, split.boxes, split.classes, class_names, 0.75)
    return {
        "photographs": len(split),
        "detection": detection,
        "average_precision_at_overlap_0.75": strict["any_defect"]["average_precision"],
        "verdicts": photograph_verdicts(
            detections, split.boxes, detection["any_defect"]["threshold"]
        ),
    }


def summary_line(label: str, result: dict) -> str:
    """Format the headline numbers of a :func:`score` result on one line."""
    any_defect = result["detection"]["any_defect"]
    return (
        f"{label}: average precision {any_defect['average_precision']:.3f}, "
        f"precision {any_defect['precision']:.2f}, recall {any_defect['recall']:.2f} "
        f"at confidence {any_defect['threshold']:.2f}; "
        f"balanced accuracy of verdicts {result['verdicts']['balanced_accuracy']:.2f}"
    )


def write_json(path: Path, content: dict) -> None:
    """Write a dictionary to a text file in JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
def command_train(arguments: argparse.Namespace) -> None:
    """Train the hand-written detector and score it on validation."""
    class_names = read_class_names(arguments.dataset)
    data = load_dataset(arguments.dataset)
    detector = GridDetector(class_names, Settings(seed=arguments.seed))

    start = time.perf_counter()
    history = detector.fit(data["train"], arguments.cache / "train.npz", report=print)
    training_seconds = time.perf_counter() - start
    detector.save(arguments.artefacts / DETECTOR_FILE)
    plots.plot_training_history(history, arguments.figures / "training-history.svg")

    detections = detector.detect(data["valid"].image_paths, arguments.cache / "valid.npz")
    result = score(detections, data["valid"], class_names)
    result.update(
        method="hand-written grid regression",
        split="valid",
        training_seconds=training_seconds,
        parameters=detector.network.number_of_parameters(),
        passes=len(history["loss"]),
    )
    write_json(arguments.artefacts / "detector-validation.json", result)
    print(summary_line("validation", result))


def command_evaluate(arguments: argparse.Namespace) -> None:
    """Score the trained detector once on the test photographs."""
    class_names = read_class_names(arguments.dataset)
    test = load_dataset(arguments.dataset)["test"]
    detector = GridDetector.load(arguments.artefacts / DETECTOR_FILE)
    validation = json.loads((arguments.artefacts / "detector-validation.json").read_text())
    threshold = validation["detection"]["any_defect"]["threshold"]

    # No saved features here, so the timing includes reading the photographs.
    start = time.perf_counter()
    detections = detector.detect(test.image_paths)
    seconds = (time.perf_counter() - start) / len(test)

    result = score(detections, test, class_names, threshold)
    result.update(
        method="hand-written grid regression",
        split="test",
        seconds_per_photograph=seconds,
        parameters=detector.network.number_of_parameters(),
        training_seconds=validation["training_seconds"],
    )
    write_json(arguments.artefacts / "detector-test.json", result)
    plots.plot_precision_recall(
        {"hand-written detector": result["detection"]["any_defect"]},
        arguments.figures / "precision-recall.svg",
    )
    # Show evenly spaced photographs, not hand-picked successes.
    shown = list(range(0, len(test), max(1, len(test) // 6)))[:6]
    plots.plot_examples(
        [test.image_paths[i] for i in shown],
        [detections[i] for i in shown],
        [test.boxes[i] for i in shown],
        threshold,
        arguments.figures / "example-detections.png",
    )
    print(summary_line("test", result))


def command_baseline(arguments: argparse.Namespace) -> None:
    """Train and score the Ultralytics alternative the same way."""
    # Imported here so the other commands do not have to load PyTorch.
    from ultralytics import YOLO

    from . import baseline

    class_names = read_class_names(arguments.dataset)
    data = load_dataset(arguments.dataset)
    if arguments.weights:
        model = YOLO(arguments.weights)
        # Ultralytics logs the running total of training time in a table
        # beside the weights folder; use it if it is there.
        log = Path(arguments.weights).resolve().parent.parent / "results.csv"
        training_seconds = None
        if log.is_file():
            header, *rows = log.read_text().strip().splitlines()
            training_seconds = float(rows[-1].split(",")[header.split(",").index("time")])
    else:
        start = time.perf_counter()
        model = baseline.train(
            arguments.dataset / "data.yaml",
            arguments.epochs,
            device=arguments.device,
            seed=arguments.seed,
            project=str(Path("runs").resolve()),
            name="baseline",
            exist_ok=True,
        )
        training_seconds = time.perf_counter() - start

    validation = score(
        baseline.detect(model, data["valid"].image_paths), data["valid"], class_names
    )
    threshold = validation["detection"]["any_defect"]["threshold"]
    start = time.perf_counter()
    detections = baseline.detect(model, data["test"].image_paths)
    seconds = (time.perf_counter() - start) / len(data["test"])

    result = score(detections, data["test"], class_names, threshold)
    for content, split in ((validation, "valid"), (result, "test")):
        content.update(
            method="Ultralytics YOLOv8 nano",
            split=split,
            parameters=int(sum(p.numel() for p in model.model.parameters())),
            training_seconds=training_seconds,
        )
    result["seconds_per_photograph"] = seconds
    write_json(arguments.artefacts / "baseline-validation.json", validation)
    write_json(arguments.artefacts / "baseline-test.json", result)
    print(summary_line("validation", validation))
    print(summary_line("test", result))


def command_compare(arguments: argparse.Namespace) -> None:
    """Put the two sets of test results side by side."""
    results = {
        "Hand-written detector": json.loads(
            (arguments.artefacts / "detector-test.json").read_text()
        ),
        "Ultralytics YOLOv8 nano": json.loads(
            (arguments.artefacts / "baseline-test.json").read_text()
        ),
    }

    def cell(value, pattern: str) -> str:
        return "not recorded" if value is None else pattern.format(value)

    rows = [
        ("Average precision, any defect (overlap 0.5)",
         lambda r: cell(r["detection"]["any_defect"]["average_precision"], "{:.3f}")),
        ("Average precision, any defect (overlap 0.75)",
         lambda r: cell(r["average_precision_at_overlap_0.75"], "{:.3f}")),
        ("Mean average precision over classes (overlap 0.5)",
         lambda r: cell(r["detection"]["mean_average_precision"], "{:.3f}")),
        ("Precision at chosen confidence",
         lambda r: cell(r["detection"]["any_defect"]["precision"], "{:.2f}")),
        ("Recall at chosen confidence",
         lambda r: cell(r["detection"]["any_defect"]["recall"], "{:.2f}")),
        ("Defective packs flagged (sensitivity)",
         lambda r: cell(r["verdicts"]["sensitivity"], "{:.2f}")),
        ("Clean packs passed (specificity)",
         lambda r: cell(r["verdicts"]["specificity"], "{:.2f}")),
        ("Balanced accuracy of verdicts",
         lambda r: cell(r["verdicts"]["balanced_accuracy"], "{:.2f}")),
        ("Learned parameters", lambda r: cell(r["parameters"], "{:,}")),
        ("Training time (minutes)",
         lambda r: cell(r["training_seconds"] and r["training_seconds"] / 60, "{:.1f}")),
        ("Time per photograph (milliseconds)",
         lambda r: cell(r["seconds_per_photograph"] * 1000, "{:.1f}")),
    ]
    lines = ["| Measure | " + " | ".join(results) + " |", "| --- |" + " ---: |" * len(results)]
    for name, value in rows:
        lines.append(f"| {name} | " + " | ".join(value(r) for r in results.values()) + " |")
    table = "\n".join(lines)
    (arguments.artefacts / "comparison.md").write_text(table + "\n", encoding="utf-8")
    plots.plot_precision_recall(
        {name: result["detection"]["any_defect"] for name, result in results.items()},
        arguments.figures / "precision-recall-comparison.svg",
    )
    print(table)


def command_study(arguments: argparse.Namespace) -> None:
    """Measure how each design choice changes the validation score."""
    from .study import run_study

    results = run_study(arguments.dataset, arguments.cache, arguments.workers)
    write_json(arguments.artefacts / "study.json", {"results": results})
    plots.plot_study(results, arguments.figures / "study.svg")


def command_profile(arguments: argparse.Namespace) -> None:
    """Time the detector and find where the time goes."""
    class_names = read_class_names(arguments.dataset)
    data = load_dataset(arguments.dataset)
    lines = []

    # Two passes are enough to see where training time goes.
    detector = GridDetector(class_names, Settings(epochs=2, seed=arguments.seed))
    profiler = cProfile.Profile()
    profiler.enable()
    detector.fit(data["train"], arguments.cache / "train.npz")
    profiler.disable()
    lines.append("Training (two passes): the ten functions with most time inside them\n")
    lines.append(_profile_table(profiler))

    photographs = data["test"].image_paths
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    detector.detect(photographs)
    profiler.disable()
    seconds = (time.perf_counter() - start) / len(photographs)
    lines.append(f"\nDetection on {len(photographs)} photographs read from disk\n")
    lines.append(_profile_table(profiler))

    network = detector.network
    cells = detector.grid.rows * detector.grid.columns
    lines.append(
        f"\nTime per photograph: {seconds * 1000:.1f} milliseconds\n"
        f"Network layers: {network.layer_sizes}\n"
        f"Learned parameters: {network.number_of_parameters():,}\n"
        f"Floating point operations per grid cell: {network.operations_per_prediction():,}\n"
        f"Floating point operations per photograph ({cells} cells): "
        f"{network.operations_per_prediction() * cells:,}\n"
    )
    report = "".join(lines)
    (arguments.artefacts / "profile.txt").write_text(report, encoding="utf-8")
    print(report)


def _profile_table(profiler: cProfile.Profile) -> str:
    """Format the ten most expensive functions recorded by a profiler."""
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("tottime").print_stats(10)
    return stream.getvalue()


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Describe the commands and their options."""
    parser = argparse.ArgumentParser(
        prog="packaging-defects",
        description="Detect pork rasher packaging defects in photographs.",
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"),
                        help="folder holding data.yaml and the train, valid and test folders")
    parser.add_argument("--artefacts", type=Path, default=Path("artefacts"),
                        help="folder for the trained detector and result files")
    parser.add_argument("--figures", type=Path, default=Path("figures"),
                        help="folder for figures")
    parser.add_argument("--cache", type=Path, default=Path("cache"),
                        help="folder for intermediate files that can be rebuilt")
    parser.add_argument("--seed", type=int, default=0, help="seed for everything random")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("train", help=command_train.__doc__).set_defaults(run=command_train)
    commands.add_parser("evaluate", help=command_evaluate.__doc__).set_defaults(
        run=command_evaluate
    )
    baseline = commands.add_parser("baseline", help=command_baseline.__doc__)
    baseline.add_argument("--epochs", type=int, default=30, help="passes through the data")
    baseline.add_argument("--device", default=None,
                          help="'mps' on Apple silicon, 'cpu', or a graphics card number")
    baseline.add_argument("--weights", default=None,
                          help="score an already trained model instead of training")
    baseline.set_defaults(run=command_baseline)
    commands.add_parser("compare", help=command_compare.__doc__).set_defaults(run=command_compare)
    study = commands.add_parser("study", help=command_study.__doc__)
    study.add_argument("--workers", type=int, default=2, help="trainings to run at once")
    study.set_defaults(run=command_study)
    commands.add_parser("profile", help=command_profile.__doc__).set_defaults(run=command_profile)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the command given on the command line."""
    arguments = build_parser().parse_args(argv)
    arguments.run(arguments)


if __name__ == "__main__":
    main()
