"""Detection and localisation of pork rasher packaging defects.

The package contains two solutions to the same problem, so that they can be
compared on evidence:

* **The main solution** is :class:`packaging_defects.detector.GridDetector`, a
  hand-written implementation of the YOLO detection process built from the
  neural network regression taught in Week 5 of MMA3001.
* **The alternative solution** is the Ultralytics YOLO library, wrapped in a
  few lines in :mod:`packaging_defects.baseline`.

Modules, in the order the data flows through them:

| Module | Purpose |
| --- | --- |
| `packaging_defects.dataset` | reads photographs and YOLO label files |
| `packaging_defects.features` | summarises each photograph as blocks |
| `packaging_defects.grid` | turns boxes into regression targets and back |
| `packaging_defects.network` | the hand-written neural network regressor |
| `packaging_defects.detector` | ties the steps above into one detector |
| `packaging_defects.boxes` | box geometry shared by several modules |
| `packaging_defects.metrics` | scores detections against the labels |
| `packaging_defects.baseline` | the Ultralytics alternative |
| `packaging_defects.study` | measures the effect of each design choice |
| `packaging_defects.plots` | figures for the report |
| `packaging_defects.command_line` | the `packaging-defects` command |
"""

__version__ = "1.0.0"
