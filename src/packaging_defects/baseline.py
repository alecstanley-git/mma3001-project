"""The alternative solution: the Ultralytics YOLO library, used as supplied.

This is deliberately only a few lines.  Everything that the rest of this
package writes out by hand (the network, the loss, the training loop and the
decoding of predictions) happens inside the library here.  Comparing the two
shows what the hand-written solution gives up, and what it costs, against a
ready-made tool.
"""

from pathlib import Path

from ultralytics import YOLO


def train(data_file: Path, epochs: int = 30, **options) -> YOLO:
    """Fine-tune a small pretrained YOLO model on the dataset.

    Parameters
    ----------
    data_file : pathlib.Path
        The dataset's ``data.yaml``.
    epochs : int
        Number of passes through the training photographs.
    **options
        Passed straight to Ultralytics, for example ``device="mps"``.

    Returns
    -------
    ultralytics.YOLO
        The trained model, holding the weights that scored best in validation.
    """
    model = YOLO("yolov8n.pt")
    model.train(data=str(data_file), epochs=epochs, **options)
    return model


def detect(model: YOLO, image_paths: list[Path]) -> list[tuple]:
    """Find defects in photographs.

    Parameters
    ----------
    model : ultralytics.YOLO
        A trained model.
    image_paths : list of pathlib.Path
        Photographs to examine.

    Returns
    -------
    list of (boxes, scores, classes)
        One entry per photograph, in the same form as
        :meth:`packaging_defects.detector.GridDetector.detect`: boxes in corner
        format with fractional coordinates, a confidence per box, and a class
        number per box.
    """
    results = model.predict([str(path) for path in image_paths], conf=0.001, verbose=False)
    return [
        (r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))
        for r in results
    ]
