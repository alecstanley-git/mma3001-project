"""Tests of the design-choice study."""

from packaging_defects import study
from packaging_defects.detector import Settings


def test_each_variation_changes_exactly_one_setting():
    chosen = Settings()
    listed = study.variations(chosen)
    labels = [label for label, _ in listed]
    assert len(set(labels)) == len(labels)
    assert listed[0][1] == chosen
    for _, settings in listed[1:]:
        changed = [
            name for name in vars(chosen) if getattr(settings, name) != getattr(chosen, name)
        ]
        assert len(changed) == 1


def test_study_trains_and_scores_every_variation(tiny_dataset, tmp_path, monkeypatch):
    quick = Settings(hidden_layers=(16,), epochs=2)
    monkeypatch.setattr(
        study, "variations", lambda chosen: [("quick", quick), ("linear", Settings(hidden_layers=(), epochs=2))]
    )
    results = study.run_study(tiny_dataset, tmp_path / "cache", workers=2)
    assert [result["label"] for result in results] == ["quick", "linear"]
    assert results[0]["parameters"] > results[1]["parameters"]
    assert all(0.0 <= result["average_precision"] <= 1.0 for result in results)
    assert (tmp_path / "cache" / "train.npz").is_file()
