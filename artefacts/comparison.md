| Measure | Hand-written detector | Ultralytics YOLOv8 nano |
| --- | ---: | ---: |
| Average precision, any defect (overlap 0.5) | 0.512 | 0.738 |
| Average precision, any defect (overlap 0.75) | 0.154 | 0.420 |
| Mean average precision over classes (overlap 0.5) | 0.201 | 0.582 |
| Precision at chosen confidence | 0.69 | 0.75 |
| Recall at chosen confidence | 0.43 | 0.68 |
| Defective packs flagged (sensitivity) | 0.69 | 0.85 |
| Clean packs passed (specificity) | 0.92 | 0.92 |
| Balanced accuracy of verdicts | 0.80 | 0.88 |
| Learned parameters | 206,410 | 3,011,823 |
| Training time (minutes) | 2.4 | 55.2 |
| Time per photograph (milliseconds) | 5.8 | 14.7 |
