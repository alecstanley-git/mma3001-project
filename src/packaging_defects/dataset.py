"""Reading the YOLO-format dataset from disk.

Expected layout (the standard Roboflow "YOLOv8" export)::

    dataset/
        data.yaml               class names
        train/images/*.jpg      training photographs
        train/labels/*.txt      one text file per photograph
        valid/...               same layout, used to choose settings
        test/...                same layout, used once for the final result

Each line of a label file describes one defect as five numbers::

    class  centre-x  centre-y  width  height

where the four coordinates are fractions of the image size (0 to 1).  An empty
label file means the photograph shows no defect.

How unusual inputs are handled
------------------------------
* A photograph with **no label file** is treated as showing no defect, which is
  the YOLO convention, and a warning is printed.
* A label line that is **malformed** (wrong number of values, a class number
  that is not in ``data.yaml``, or coordinates outside 0 to 1) raises
  ``ValueError`` naming the file, because silently skipping it would hide a
  labelling problem.
* A photograph that **cannot be opened** raises ``OSError`` naming the file.
* A photograph of a **different size** is resized to the working size, so the
  model can still be applied to it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

#: Names of the three parts of the dataset and the job each one does.
SPLIT_NAMES = ("train", "valid", "test")

#: Width and height, in pixels, of the photographs in the dataset.
FULL_IMAGE_SIZE = (720, 540)

#: Each photograph is shrunk by this factor before any further processing.
REDUCTION = 2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Split:
    """The photographs and labels of one part of the dataset.

    Attributes
    ----------
    name : str
        ``"train"``, ``"valid"`` or ``"test"``.
    image_paths : list of pathlib.Path
        Where each photograph came from.
    boxes : list of numpy.ndarray
        One array per photograph, shape (defects, 4), in centre format with
        fractional coordinates (see :mod:`packaging_defects.boxes`).
    classes : list of numpy.ndarray
        One integer array per photograph giving the class of each defect.
    sources : list of str
        Name of the original video frame each photograph was made from.
    """

    name: str
    image_paths: list[Path]
    boxes: list[np.ndarray]
    classes: list[np.ndarray]
    sources: list[str]

    def __len__(self) -> int:
        return len(self.image_paths)

    def subset(self, indices) -> "Split":
        """Return a new split containing only the photographs at ``indices``."""
        indices = [int(i) for i in indices]
        return Split(
            name=self.name,
            image_paths=[self.image_paths[i] for i in indices],
            boxes=[self.boxes[i] for i in indices],
            classes=[self.classes[i] for i in indices],
            sources=[self.sources[i] for i in indices],
        )


def read_class_names(dataset_directory: Path) -> list[str]:
    """Read the defect class names from ``data.yaml``.

    Parameters
    ----------
    dataset_directory : pathlib.Path
        Folder that contains ``data.yaml``.

    Returns
    -------
    list of str
        Class names; position in the list is the class number in label files.

    Raises
    ------
    FileNotFoundError
        If ``data.yaml`` is missing.
    ValueError
        If the file has no ``names`` entry.
    """
    path = Path(dataset_directory) / "data.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Unpack the Roboflow YOLOv8 export into {dataset_directory}."
        )
    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    names = settings.get("names")
    if not names:
        raise ValueError(f"{path} does not list any class names")
    # YOLO allows either a list or a {number: name} mapping.
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    return [str(name) for name in names]


def read_label_file(path: Path, number_of_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the defects labelled in one photograph.

    Parameters
    ----------
    path : pathlib.Path
        The ``.txt`` label file.
    number_of_classes : int
        How many classes exist; used to reject impossible class numbers.

    Returns
    -------
    classes : numpy.ndarray of int, shape (defects,)
    boxes : numpy.ndarray of float, shape (defects, 4)
        Centre format, fractional coordinates.

    Raises
    ------
    ValueError
        If any line is malformed.  The message names the file and line.
    """
    classes, boxes = [], []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"{path}, line {line_number}: expected 5 values, found {len(values)}")
        try:
            class_number = int(values[0])
            box = [float(value) for value in values[1:]]
        except ValueError as error:
            raise ValueError(f"{path}, line {line_number}: values must be numbers") from error
        if not 0 <= class_number < number_of_classes:
            raise ValueError(f"{path}, line {line_number}: unknown class {class_number}")
        if min(box) < 0.0 or max(box) > 1.0 or box[2] <= 0.0 or box[3] <= 0.0:
            raise ValueError(f"{path}, line {line_number}: box must lie inside the image")
        classes.append(class_number)
        boxes.append(box)
    return np.array(classes, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


def source_frame_name(file_name: str) -> str:
    """Recover the name of the original video frame from a Roboflow file name.

    Roboflow saved three brightness-adjusted copies of each training frame and
    named them like ``frame_with_red_3_jpg.rf.<random letters>.jpg``.  The part
    before ``.rf.`` identifies the frame the copy was made from, which is what
    is needed to keep copies of one frame together (see
    :func:`remove_frames_seen_elsewhere`).

    Parameters
    ----------
    file_name : str
        File name of a photograph or label file.

    Returns
    -------
    str
        The original frame name, for example ``"frame_with_red_3"``.
    """
    name = Path(file_name).name
    name = name.split(".rf.")[0] if ".rf." in name else Path(name).stem
    return name.removesuffix("_jpg")


def load_image(path: Path, reduction: int = REDUCTION) -> np.ndarray:
    """Open one photograph and shrink it by averaging blocks of pixels.

    Averaging every ``reduction`` by ``reduction`` block of pixels into one
    pixel shrinks the picture without the speckled artefacts that simply
    skipping pixels would give, and makes every later step faster.

    Parameters
    ----------
    path : pathlib.Path
        Image file to open.
    reduction : int
        Shrink factor; must divide both sides of :data:`FULL_IMAGE_SIZE`.

    Returns
    -------
    numpy.ndarray, shape (height, width, 3), dtype uint8
        The reduced photograph in red-green-blue order.

    Raises
    ------
    OSError
        If the file is missing or is not a readable image.
    """
    width, height = FULL_IMAGE_SIZE
    if width % reduction or height % reduction:
        raise ValueError(f"reduction {reduction} must divide the image size {FULL_IMAGE_SIZE}")
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != FULL_IMAGE_SIZE:
            # Unsupported sizes are stretched to the working size. Labels are
            # fractions of the image, so they stay correct.
            image = image.resize(FULL_IMAGE_SIZE, Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32)
    # Add up the pixels of each block by taking every `reduction`-th pixel,
    # starting from each position within a block in turn.  This gives the same
    # answer as reshaping into blocks and averaging, but profiling showed it
    # to be several times faster because each step reads memory in order.
    total = np.zeros((height // reduction, width // reduction, 3), dtype=np.float32)
    for down in range(reduction):
        for across in range(reduction):
            total += pixels[down::reduction, across::reduction]
    return np.round(total / (reduction * reduction)).astype(np.uint8)


def load_images(image_paths: list[Path], reduction: int = REDUCTION) -> np.ndarray:
    """Open several photographs and stack them into one array.

    Parameters
    ----------
    image_paths : list of pathlib.Path
    reduction : int
        Shrink factor passed to :func:`load_image`.

    Returns
    -------
    numpy.ndarray, shape (n, height, width, 3), dtype uint8

    Raises
    ------
    OSError
        If any file cannot be opened; the message names the file.
    """
    images = []
    for path in image_paths:
        try:
            images.append(load_image(path, reduction))
        except Exception as error:
            # Every kind of failure is reported the same way.  (Once the
            # Ultralytics library has been imported it replaces the image
            # opener with one that raises other error types.)
            raise OSError(f"Cannot read photograph {path}: {error}") from error
    return np.stack(images)


def load_split(dataset_directory: Path, name: str) -> Split:
    """Find every photograph of one part of the dataset and read its labels.

    Parameters
    ----------
    dataset_directory : pathlib.Path
        Folder containing ``data.yaml`` and the split folders.
    name : str
        ``"train"``, ``"valid"`` or ``"test"``.

    Returns
    -------
    Split

    Raises
    ------
    FileNotFoundError
        If the image folder does not exist or is empty.
    ValueError
        If a label file is malformed.
    """
    dataset_directory = Path(dataset_directory)
    image_directory = dataset_directory / name / "images"
    label_directory = dataset_directory / name / "labels"
    paths = []
    if image_directory.is_dir():
        paths = sorted(
            path for path in image_directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not paths:
        raise FileNotFoundError(f"No photographs found in {image_directory}")
    number_of_classes = len(read_class_names(dataset_directory))

    boxes, classes = [], []
    for path in paths:
        label_path = label_directory / f"{path.stem}.txt"
        if label_path.is_file():
            image_classes, image_boxes = read_label_file(label_path, number_of_classes)
        else:
            warnings.warn(f"No label file for {path.name}; treating it as defect-free")
            image_classes, image_boxes = np.zeros(0, dtype=int), np.zeros((0, 4))
        boxes.append(image_boxes)
        classes.append(image_classes)

    return Split(
        name=name,
        image_paths=paths,
        boxes=boxes,
        classes=classes,
        sources=[source_frame_name(path.name) for path in paths],
    )


def remove_frames_seen_elsewhere(training: Split, *held_out: Split) -> Split:
    """Drop training photographs whose original frame is also held out.

    Roboflow made its brightness-adjusted copies *before* dividing the frames
    into training, validation and test parts, so a few frames have one copy in
    training and another in validation or test.  Scoring a model on a frame it
    has effectively already seen would flatter it, so those training copies
    are removed.

    Parameters
    ----------
    training : Split
        The training part.
    *held_out : Split
        The validation and test parts.

    Returns
    -------
    Split
        The training part without the shared frames.
    """
    seen = {source for split in held_out for source in split.sources}
    keep = [index for index, source in enumerate(training.sources) if source not in seen]
    return training.subset(keep)


def load_dataset(dataset_directory: Path) -> dict[str, Split]:
    """Read all three parts of the dataset, with shared frames removed.

    Parameters
    ----------
    dataset_directory : pathlib.Path
        Folder containing ``data.yaml`` and the split folders.

    Returns
    -------
    dict of str to Split
        Keys ``"train"``, ``"valid"`` and ``"test"``.
    """
    splits = {name: load_split(dataset_directory, name) for name in SPLIT_NAMES}
    splits["train"] = remove_frames_seen_elsewhere(splits["train"], splits["valid"], splits["test"])
    return splits
