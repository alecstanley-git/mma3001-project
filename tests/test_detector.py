"""Tests of the complete hand-written detector on a small made-up dataset."""

from dataclasses import replace

import numpy as np
import pytest

from packaging_defects.dataset import load_dataset, read_class_names
from packaging_defects.detector import GridDetector, Settings
from packaging_defects.metrics import evaluate_detections

QUICK = Settings(hidden_layers=(48,), epochs=30, batch_size=128, learning_rate=0.002, patience=30)


@pytest.fixture(scope="module")
def trained(tiny_dataset):
    data = load_dataset(tiny_dataset)
    detector = GridDetector(read_class_names(tiny_dataset), QUICK)
    detector.fit(data["train"])
    return detector, data


def test_detector_finds_the_patches_in_unseen_photographs(trained):
    detector, data = trained
    test = data["test"]
    detections = detector.detect(test.image_paths)
    result = evaluate_detections(detections, test.boxes, test.classes, detector.class_names)
    assert result["any_defect"]["average_precision"] > 0.8


def test_training_history_is_recorded(trained):
    detector, _ = trained
    assert len(detector.history["loss"]) == len(detector.history["validation_score"])
    assert detector.history["loss"][-1] < detector.history["loss"][0]


def test_detections_have_the_documented_form(trained):
    detector, data = trained
    boxes, scores, classes = detector.detect(data["test"].image_paths[:1])[0]
    assert boxes.shape == (len(scores), 4) and classes.shape == scores.shape
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert np.all((boxes >= 0.0) & (boxes <= 1.0))
    assert np.all(np.diff(scores) <= 0)  # most confident first


def test_input_length_matches_the_documented_count(trained):
    detector, _ = trained
    assert detector.network.layer_sizes[0] == detector.number_of_inputs()


def test_saved_detector_gives_identical_detections(trained, tmp_path):
    detector, data = trained
    detector.save(tmp_path / "detector.pkl")
    restored = GridDetector.load(tmp_path / "detector.pkl")
    assert restored.settings == detector.settings
    paths = data["test"].image_paths[:4]
    for first, second in zip(detector.detect(paths), restored.detect(paths)):
        assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_untrained_detector_refuses_to_run(tiny_dataset, tmp_path):
    detector = GridDetector(["a"])
    with pytest.raises(RuntimeError):
        detector.detect(load_dataset(tiny_dataset)["test"].image_paths[:1])
    with pytest.raises(RuntimeError):
        detector.save(tmp_path / "detector.pkl")


def test_detecting_in_no_photographs_returns_nothing(trained):
    assert trained[0].detect([]) == []


def test_copies_of_one_frame_stay_together_when_holding_back(tiny_dataset, monkeypatch):
    """The photographs set aside for the stopping rule must not share a frame with the rest."""
    import packaging_defects.detector as module

    seen = {}
    original = module.train

    def spy(network, make_batch, number_of_samples, **options):
        seen["samples"] = number_of_samples
        return original(network, make_batch, number_of_samples, **options)

    monkeypatch.setattr(module, "train", spy)
    data = load_dataset(tiny_dataset)
    settings = replace(QUICK, epochs=1, mirrored_copies=False, validation_fraction=0.25)
    detector = GridDetector(read_class_names(tiny_dataset), settings)
    detector.fit(data["train"])
    # 60 frames, 2 copies each: holding back 15 whole frames leaves 45 * 2 photographs.
    assert seen["samples"] == 45 * 2 * detector.grid.rows * detector.grid.columns
