"""Shared set-up for the tests: a small made-up dataset in the YOLO layout."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from packaging_defects.dataset import FULL_IMAGE_SIZE

# Stop the Ultralytics library from trying to install extra packages by itself.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

CLASS_NAMES = ["bright-patch", "unused-class"]


def write_photograph(path: Path, box: tuple | None, generator: np.random.Generator) -> None:
    """Save a dull, slightly noisy photograph with an optional bright striped patch.

    ``box`` is ``(centre x, centre y, width, height)`` as fractions of the
    image, the same form as a YOLO label.
    """
    width, height = FULL_IMAGE_SIZE
    pixels = generator.normal(90.0, 6.0, (height, width, 3))
    if box is not None:
        centre_x, centre_y, box_width, box_height = box
        left, right = int((centre_x - box_width / 2) * width), int((centre_x + box_width / 2) * width)
        top, bottom = int((centre_y - box_height / 2) * height), int((centre_y + box_height / 2) * height)
        pixels[top:bottom, left:right] = 215.0
        pixels[top:bottom:8, left:right] = 40.0  # dark stripes give the patch strong edges
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(path, quality=92)


def build_dataset(root: Path, frames: dict[str, int], seed: int = 0) -> Path:
    """Create a dataset folder with ``frames[split]`` original frames in each split.

    Training frames are written as two copies each, named the way Roboflow
    names its brightness-adjusted copies.  Every fourth frame has no defect.
    """
    generator = np.random.default_rng(seed)
    (root / "data.yaml").write_text(f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    for split, count in frames.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for frame in range(count):
            box = None
            if frame % 4 != 3:
                box = (
                    generator.uniform(0.2, 0.8),
                    generator.uniform(0.2, 0.8),
                    generator.uniform(0.14, 0.22),
                    generator.uniform(0.18, 0.28),
                )
            for copy in range(2 if split == "train" else 1):
                stem = f"{split}_frame_{frame}_jpg.rf.copy{copy}"
                write_photograph(root / split / "images" / f"{stem}.jpg", box, generator)
                # Like the real export, the last line has no line ending.
                label = "" if box is None else "0 " + " ".join(f"{value:.6f}" for value in box)
                (root / split / "labels" / f"{stem}.txt").write_text(label)
    return root


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> Path:
    """A made-up dataset big enough to train a detector in a few seconds."""
    return build_dataset(
        tmp_path_factory.mktemp("dataset"), {"train": 60, "valid": 16, "test": 16}
    )
