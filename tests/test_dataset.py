"""Tests of reading the YOLO-format dataset, including bad input."""

import numpy as np
import pytest
from PIL import Image

from packaging_defects.dataset import (
    FULL_IMAGE_SIZE,
    REDUCTION,
    load_dataset,
    load_image,
    load_images,
    load_split,
    read_class_names,
    read_label_file,
    source_frame_name,
)


def test_class_names_are_read(tiny_dataset):
    assert read_class_names(tiny_dataset) == ["bright-patch", "unused-class"]


def test_class_names_given_as_a_numbered_mapping(tmp_path):
    (tmp_path / "data.yaml").write_text("names:\n  1: second\n  0: first\n")
    assert read_class_names(tmp_path) == ["first", "second"]


def test_missing_settings_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_class_names(tmp_path)


def test_settings_file_without_names_is_rejected(tmp_path):
    (tmp_path / "data.yaml").write_text("nc: 2\n")
    with pytest.raises(ValueError):
        read_class_names(tmp_path)


def test_label_file_without_final_line_ending(tmp_path):
    path = tmp_path / "label.txt"
    path.write_text("1 0.5 0.5 0.2 0.2\n0 0.25 0.75 0.1 0.3")
    classes, boxes = read_label_file(path, number_of_classes=2)
    assert classes.tolist() == [1, 0]
    assert np.allclose(boxes[1], [0.25, 0.75, 0.1, 0.3])


def test_empty_label_file_means_no_defects(tmp_path):
    path = tmp_path / "label.txt"
    path.write_text("")
    classes, boxes = read_label_file(path, number_of_classes=2)
    assert classes.shape == (0,) and boxes.shape == (0, 4)


@pytest.mark.parametrize(
    "line",
    [
        "0 0.5 0.5 0.2",  # too few values
        "0 0.5 0.5 0.2 0.2 0.1",  # too many values
        "7 0.5 0.5 0.2 0.2",  # class that does not exist
        "0 1.5 0.5 0.2 0.2",  # centre outside the image
        "0 0.5 0.5 0.0 0.2",  # no width
        "0 0.5 half 0.2 0.2",  # not a number
    ],
)
def test_malformed_label_lines_are_rejected(tmp_path, line):
    path = tmp_path / "label.txt"
    path.write_text(line)
    with pytest.raises(ValueError, match="label.txt"):
        read_label_file(path, number_of_classes=2)


def test_source_frame_name():
    name = "frame_with_red_3_jpg.rf.45f097aec54556d346dc347924c4a9a0.jpg"
    assert source_frame_name(name) == "frame_with_red_3"
    assert source_frame_name("plain_photo.png") == "plain_photo"


def test_load_image_shrinks_by_averaging(tmp_path):
    width, height = FULL_IMAGE_SIZE
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, ::2] = 200  # alternating dark and bright columns average to 100
    path = tmp_path / "stripes.png"
    Image.fromarray(pixels).save(path)
    image = load_image(path)
    assert image.shape == (height // REDUCTION, width // REDUCTION, 3)
    assert np.all(image == 100)


@pytest.mark.parametrize("reduction", [1, 2, 4])
def test_load_image_matches_plain_block_averaging(tiny_dataset, reduction):
    """The fast block sum must give exactly the pixels of the obvious method."""
    path = load_split(tiny_dataset, "valid").image_paths[0]
    width, height = FULL_IMAGE_SIZE
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    blocks = pixels.reshape(height // reduction, reduction, width // reduction, reduction, 3)
    expected = np.round(blocks.mean(axis=(1, 3))).astype(np.uint8)
    assert np.array_equal(load_image(path, reduction), expected)


def test_load_image_resizes_other_sizes(tmp_path):
    path = tmp_path / "small.png"
    Image.new("L", (300, 200), 128).save(path)  # also a different colour mode
    image = load_image(path)
    assert image.shape == (FULL_IMAGE_SIZE[1] // REDUCTION, FULL_IMAGE_SIZE[0] // REDUCTION, 3)


def test_unreadable_photograph_is_reported(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_text("this is not a photograph")
    with pytest.raises(OSError, match="broken.jpg"):
        load_images([path])


def test_missing_split_folder_is_reported(tiny_dataset):
    with pytest.raises(FileNotFoundError):
        load_split(tiny_dataset, "no-such-split")


def test_load_split_reads_photographs_and_labels(tiny_dataset):
    split = load_split(tiny_dataset, "valid")
    assert len(split) == 16
    assert sum(len(boxes) for boxes in split.boxes) == 12  # every fourth frame is clean
    assert split.sources[0].startswith("valid_frame_")


def test_photograph_without_label_file_counts_as_clean(tmp_path):
    (tmp_path / "data.yaml").write_text("names: [a]\n")
    (tmp_path / "train" / "images").mkdir(parents=True)
    Image.new("RGB", FULL_IMAGE_SIZE).save(tmp_path / "train" / "images" / "photo.jpg")
    with pytest.warns(UserWarning, match="No label file"):
        split = load_split(tmp_path, "train")
    assert split.boxes[0].shape == (0, 4)


def test_training_copies_of_held_out_frames_are_removed(tmp_path):
    (tmp_path / "data.yaml").write_text("names: [a]\n")
    files = {
        "train": ["frame_1_jpg.rf.a.jpg", "frame_1_jpg.rf.b.jpg", "frame_2_jpg.rf.a.jpg"],
        "valid": ["frame_2_jpg.rf.z.jpg"],
        "test": ["frame_3_jpg.rf.z.jpg"],
    }
    for split, names in files.items():
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        for name in names:
            Image.new("RGB", (8, 8)).save(tmp_path / split / "images" / name)
            (tmp_path / split / "labels" / name.replace(".jpg", ".txt")).write_text("")
    data = load_dataset(tmp_path)
    assert sorted(set(data["train"].sources)) == ["frame_1"]
    assert len(data["train"]) == 2


def test_subset_keeps_matching_entries(tiny_dataset):
    split = load_split(tiny_dataset, "valid")
    part = split.subset([3, 5])
    assert part.image_paths == [split.image_paths[3], split.image_paths[5]]
    assert part.sources == [split.sources[3], split.sources[5]]
