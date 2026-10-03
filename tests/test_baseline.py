"""Test of the thin wrapper round the Ultralytics library.

Training a YOLO model takes many minutes, so it is not done here.  The test
checks the one piece of logic in the wrapper: converting the library's results
to the form shared with the hand-written detector.
"""

import numpy as np


def test_baseline_detections_are_converted_to_the_shared_form():
    """Check the conversion without training: a stand-in replaces the YOLO model."""
    # Imported here because loading Ultralytics is slow and it changes how
    # image files are opened for the rest of the session.
    from packaging_defects import baseline

    class Tensor:
        def __init__(self, values):
            self.values = np.array(values)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Boxes:
        xyxyn = Tensor([[0.1, 0.2, 0.3, 0.4]])
        conf = Tensor([0.75])
        cls = Tensor([3.0])

    class Result:
        boxes = Boxes()

    class Model:
        def predict(self, paths, conf, verbose):
            return [Result() for _ in paths]

    (boxes, scores, classes), = baseline.detect(Model(), ["photo.jpg"])
    assert boxes.tolist() == [[0.1, 0.2, 0.3, 0.4]]
    assert scores.tolist() == [0.75]
    assert classes.tolist() == [3] and classes.dtype.kind == "i"
