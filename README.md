# MMA3001 Project

### by Alec Stanley

The chosen project is Dataset 3 - "Automated Detection of Pork Rasher Packaging Errors and Meat Quality"

## Stage 1 — packaging-defect classification

`main.py` trains a small convolutional neural network that decides whether a
conveyor-belt frame shows a packaging defect. Everything from the image
resampling to the ROC integral is written directly against NumPy so each
numerical step is visible: there is no deep-learning framework in the
dependency list.

### Running it

```sh
uv run main.py                 # full run: ~1 min of setup, ~5 s per epoch
uv run main.py --help          # epochs, batch size, learning rate, seed, ...
uv run main.py --rebuild-cache # re-decode the JPEGs after changing preprocessing
```

The first run decodes all 3549 JPEGs and caches the decimated arrays under
`cache/` (git-ignored); later runs start in a second or two.

### What it does

| Stage | Method |
| --- | --- |
| Preprocess | 720×540 RGB → 80×60 by exact 9×9 box averaging (anti-aliases the conveyor mesh), then per-channel standardisation using training statistics only |
| Label | An image is *defective* if its COCO record carries ≥1 annotation of any defect class |
| Model | 3 × (3×3 convolution → ReLU → 2×2 max pool) → dense(32) → ReLU → dense(1); 77,777 parameters, 7.75 MFLOP per forward pass |
| Loss | Class-weighted binary cross-entropy in log-sum-exp form, with the negative class up-weighted ×3.84 to offset the 79 % positive prior |
| Optimiser | Adam with bias correction and decoupled weight decay, written out explicitly |
| Verification | Backpropagation checked against central finite differences over a sweep of step sizes |
| Selection | Weights from the best validation epoch; decision threshold fitted on validation by Youden's J |

### Results

Held-out test split (80 frames, 68 defective), threshold fitted on validation:

| | DefectNet | Always-defect baseline |
| --- | --- | --- |
| Balanced accuracy | **0.777** | 0.500 |
| ROC AUC | **0.859** | 0.543 |
| Recall (defect) | 0.721 | 1.000 |
| Specificity (clean) | 0.833 | 0.000 |
| Accuracy | 0.738 | 0.850 |

Plain accuracy is the wrong headline here: 85 % of the test frames are
defective, so the trivial "always defect" classifier beats the network on
accuracy while being useless. Balanced accuracy and AUC are the figures that
separate them.

The gradient check bottoms out at a median relative error of 1.2 × 10⁻¹⁰ near
h = 10⁻⁴·⁵, with the expected truncation and round-off branches either side.

### Outputs

* `artefacts/model.npz` — weights, decision threshold, and the standardisation statistics
* `artefacts/results.json` — full metrics, training history and timings
* `figures/` — training history, gradient-check sweep, ROC curve, confusion matrix (PDF, sized for `report.typ`)

### Known limitations

* The frames come from a continuous video, so neighbouring frames are nearly
  identical and the published train/valid/test split is not independent.
  Training frames whose source image also appears in valid or test are dropped
  (36 of them), but temporal correlation remains and the held-out scores are
  optimistic.
* The evaluation splits are small — 12 and 14 clean frames — so the specificity
  estimates have wide confidence intervals.
* Many frames show only part of a pack, and defect annotations are more likely
  on frames where more of the pack is visible. The classifier may be picking up
  some of that framing confound rather than the defect itself.
* The model answers *whether* a defect is present, not *where*. Localisation is
  the next stage.
