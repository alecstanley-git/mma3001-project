"""Tests that the commands run from start to finish and write their files."""

import json

import pytest

from packaging_defects import command_line


@pytest.fixture(scope="module")
def workspace(tiny_dataset, tmp_path_factory):
    """Run ``train`` then ``evaluate`` once and share the output folder."""
    root = tmp_path_factory.mktemp("workspace")
    common = [
        "--dataset", str(tiny_dataset),
        "--artefacts", str(root / "artefacts"),
        "--figures", str(root / "figures"),
        "--cache", str(root / "cache"),
    ]
    command_line.main([*common, "train"])
    command_line.main([*common, "evaluate"])
    return root, common


def test_train_writes_the_detector_and_validation_results(workspace):
    root, _ = workspace
    assert (root / "artefacts" / "detector.pkl").is_file()
    assert (root / "figures" / "training-history.svg").is_file()
    result = json.loads((root / "artefacts" / "detector-validation.json").read_text())
    assert result["split"] == "valid" and result["photographs"] == 16


def test_evaluate_uses_the_threshold_chosen_on_validation(workspace):
    root, _ = workspace
    validation = json.loads((root / "artefacts" / "detector-validation.json").read_text())
    test = json.loads((root / "artefacts" / "detector-test.json").read_text())
    assert test["split"] == "test"
    assert (
        test["detection"]["any_defect"]["threshold"]
        == validation["detection"]["any_defect"]["threshold"]
    )
    assert test["seconds_per_photograph"] > 0
    assert (root / "figures" / "precision-recall.svg").is_file()
    assert (root / "figures" / "example-detections.png").is_file()


def test_compare_writes_a_table(workspace, capsys):
    root, common = workspace
    # Stand in for the Ultralytics results with a copy of our own.
    result = json.loads((root / "artefacts" / "detector-test.json").read_text())
    (root / "artefacts" / "baseline-test.json").write_text(json.dumps(result))
    command_line.main([*common, "compare"])
    table = (root / "artefacts" / "comparison.md").read_text()
    assert "Average precision" in table and "Ultralytics" in table
    assert (root / "figures" / "precision-recall-comparison.svg").is_file()


def test_profile_reports_timings_and_operation_counts(workspace):
    root, common = workspace
    command_line.main([*common, "profile"])
    report = (root / "artefacts" / "profile.txt").read_text()
    assert "Time per photograph" in report and "Floating point operations" in report


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        command_line.main([])
