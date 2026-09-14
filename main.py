"""Packaging-defect classification for conveyor-borne pork rasher packs.

Stage 1 of the MMA3001 project: a binary image classifier that answers the
question *"does this frame show a packaging defect?"*.  The classifier is a
small convolutional neural network written directly on top of NumPy so that
every numerical step -- image resampling, feature standardisation, weight
initialisation, convolution, the loss, the optimiser and the evaluation
integrals -- is visible and auditable rather than hidden inside a deep-learning
framework.

Top-level input
---------------
The Roboflow export of the *Pork Rasher Error (Packaging)* dataset, laid out as
``data/{train,valid,test}/`` with one COCO annotation file per split.  Images
are 8-bit RGB JPEGs of 720x540 px showing a black conveyor mesh with one or
more transparent-lidded rasher packs partially in frame.

Top-level output
----------------
* A trained set of network weights (``artefacts/model.npz``).
* A defect probability in [0, 1] for every image of the held-out test split,
  thresholded into the labels ``defect`` / ``no defect``.
* Classification metrics (confusion matrix, balanced accuracy, precision,
  recall, F1, ROC AUC) printed to stdout and written to ``artefacts/``.
* Report figures (training curves, ROC curve, confusion matrix) in
  ``figures/``.

Label definition
----------------
The COCO file annotates defect *regions* (``packaging-error``, ``unsealed``,
``wrinkle``, ``loose-meat``, ``twisted-meat``).  For this binary stage an image
is labelled ``1`` (defective) if it carries at least one annotation of any
defect class and ``0`` otherwise.  The localisation boxes are deliberately
ignored here; they are the input to the later localisation stage.

Run with ``python main.py`` (see ``--help`` for the tunable settings).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Figures are written to disk, never shown interactively.

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the "science" matplotlib style)
from PIL import Image, UnidentifiedImageError

# ======================================================================
#  0.  Report figure style
# ======================================================================
# Figure width in inches, matched to report.typ so figure text renders at the
# same physical point size as the report body text.
#   report text width = 6.2992 in  (A4, typst default margins)
#   plotWidth         = 70%        -> 4.4094 in
# For a figure inside a #columns(2)[...] block use twoColPlotWidth instead:
#   column width = (6.2992 - 0.2520 gutter) / 2 = 3.0236 in
#   twoColPlotWidth = 80%                       -> 2.4189 in
FIG_WIDTH = 4.4094

plt.style.use("science")
plt.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_WIDTH * 2.625 / 3.5)


# ======================================================================
#  1.  Configuration
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
FIGURE_DIR = PROJECT_ROOT / "figures"
ARTEFACT_DIR = PROJECT_ROOT / "artefacts"

SPLITS = ("train", "valid", "test")
ANNOTATION_FILE = "_annotations.coco.json"

# Native frame geometry produced by the Roboflow export.  Anything else is
# rejected by the loader rather than silently rescaled, because the box-average
# decimation below assumes these exact dimensions.
NATIVE_W, NATIVE_H = 720, 540

# Decimation factor for the box (area-average) low-pass filter.  720 and 540 are
# both exactly divisible by 9, so every output pixel is the unweighted mean of a
# disjoint 9x9 block and no interpolation or edge handling is required.
DECIMATION = 9
IMG_W, IMG_H = NATIVE_W // DECIMATION, NATIVE_H // DECIMATION  # 80 x 60
IMG_C = 3  # RGB retained: pack lids are specular, meat is red, belt is black.

# Network shape.  Three convolution blocks, each halving the spatial grid, give
# a receptive field of roughly 22 px at 80x60, i.e. ~200 px in the original
# frame -- the scale of the annotated unsealed-seam and wrinkle regions.
CONV_CHANNELS = (8, 16, 32)
KERNEL = 3
DENSE_UNITS = 32

# Optimiser defaults (overridable on the command line).
DEFAULT_EPOCHS = 30
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
DEFAULT_L2 = 1e-4
DEFAULT_SEED = 20260914

# Adam constants (Kingma & Ba, 2015).
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8

DTYPE = np.float32  # Storage/compute precision for activations and weights.


# ======================================================================
#  2.  Data loading and preprocessing
# ======================================================================
def read_split_index(split: str) -> tuple[list[Path], np.ndarray, list[str]]:
    """Read one COCO split and reduce it to a binary image-level index.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"valid"``, ``"test"``.

    Returns
    -------
    paths : list[Path]
        Absolute path of every image file listed in the annotation file.
    labels : ndarray of int8, shape (n,)
        ``1`` where the image carries at least one defect annotation, else ``0``.
    sources : list[str]
        Original (pre-augmentation) file name of each image.  Roboflow writes
        three exposure-jittered copies of every source frame into ``train``, so
        this is what identifies duplicated content across splits.

    Raises
    ------
    FileNotFoundError
        If the split directory or its annotation file is absent.
    """
    split_dir = DATA_DIR / split
    annotation_path = split_dir / ANNOTATION_FILE
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Missing COCO annotations at {annotation_path}. Expected the "
            f"Roboflow export to be unpacked into {DATA_DIR}."
        )

    with annotation_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    # An image is positive if *any* annotation references it.  Category 0
    # ("packaging-error", supercategory "none") is Roboflow's placeholder root
    # node and never appears on an annotation, so no filtering is needed.
    defective_ids = {annotation["image_id"] for annotation in coco["annotations"]}

    paths, labels, sources = [], [], []
    for record in coco["images"]:
        paths.append(split_dir / record["file_name"])
        labels.append(1 if record["id"] in defective_ids else 0)
        # "extra"/"name" holds the pre-augmentation file name; fall back to the
        # exported name with the Roboflow hash suffix stripped.
        extra = record.get("extra", {})
        sources.append(
            extra.get("name") or re.sub(r"\.rf\.[0-9a-f]+", "", record["file_name"])
        )

    return paths, np.asarray(labels, dtype=np.int8), sources


def box_decimate(frame: np.ndarray, factor: int = DECIMATION) -> np.ndarray:
    """Anti-aliased integer-factor downsample by disjoint block averaging.

    Decimating by simple subsampling would alias the conveyor mesh -- a strong,
    almost periodic texture -- into the low-resolution image.  Averaging each
    ``factor x factor`` block first applies a box low-pass filter whose first
    zero sits at the new Nyquist frequency, which suppresses that aliasing at
    the cost of a mild loss of edge contrast.

    Parameters
    ----------
    frame : ndarray, shape (H, W, C)
        Image with ``H`` and ``W`` exact multiples of ``factor``.
    factor : int
        Block edge length.

    Returns
    -------
    ndarray of float32, shape (H // factor, W // factor, C)
    """
    height, width, channels = frame.shape
    blocks = frame.reshape(height // factor, factor, width // factor, factor, channels)
    return blocks.mean(axis=(1, 3), dtype=np.float32)


def load_image(path: Path) -> np.ndarray | None:
    """Load one JPEG and reduce it to a ``(C, H, W)`` float array in [0, 255].

    Returns ``None`` (with a warning) instead of raising when a file is missing,
    truncated, or not of the expected native geometry, so that a single corrupt
    frame cannot abort a multi-thousand-image ingest.
    """
    try:
        with Image.open(path) as handle:
            handle = handle.convert("RGB")  # Drops any alpha/palette encoding.
            if handle.size != (NATIVE_W, NATIVE_H):
                print(
                    f"  [skip] {path.name}: size {handle.size} != {(NATIVE_W, NATIVE_H)}"
                )
                return None
            frame = np.asarray(handle, dtype=np.float32)
    except (FileNotFoundError, OSError, UnidentifiedImageError) as error:
        print(f"  [skip] {path.name}: unreadable ({error})")
        return None

    return np.ascontiguousarray(box_decimate(frame).transpose(2, 0, 1))


def build_split_array(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Decode and decimate every image of a split into one contiguous array.

    Returns
    -------
    images : ndarray of uint8, shape (n, C, H, W)
        The decimated frames, rounded to 8-bit.  The block mean of 8-bit inputs
        is real-valued, so rounding costs at most 0.5/255 = 0.2 % per pixel --
        far below the JPEG compression noise already present -- and makes the
        on-disk cache four times smaller than a float32 one.
    labels : ndarray of int8, shape (n,)
    sources : list[str]
    """
    paths, labels, sources = read_split_index(split)
    print(f"  decoding {len(paths)} images from data/{split} ...")

    kept_images, kept_labels, kept_sources = [], [], []
    for path, label, source in zip(paths, labels, sources):
        image = load_image(path)
        if image is None:
            continue
        kept_images.append(np.rint(image).astype(np.uint8))
        kept_labels.append(label)
        kept_sources.append(source)

    if not kept_images:
        raise RuntimeError(f"No readable images in data/{split}.")

    return (
        np.stack(kept_images),
        np.asarray(kept_labels, dtype=np.int8),
        kept_sources,
    )


def load_dataset(rebuild: bool = False) -> dict[str, dict]:
    """Load all three splits, using an on-disk cache of the decoded arrays.

    Decoding 3550 JPEGs takes far longer than a training epoch, so the decimated
    arrays are cached as compressed ``.npz`` files keyed by split.  Pass
    ``rebuild=True`` to force a re-decode after changing the preprocessing.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    dataset = {}

    for split in SPLITS:
        cache_path = CACHE_DIR / f"{split}_{IMG_W}x{IMG_H}.npz"
        if cache_path.is_file() and not rebuild:
            with np.load(cache_path, allow_pickle=False) as bundle:
                images = bundle["images"]
                labels = bundle["labels"]
                sources = list(bundle["sources"])
            print(f"  loaded data/{split} from cache ({images.shape[0]} images)")
        else:
            images, labels, sources = build_split_array(split)
            np.savez_compressed(
                cache_path, images=images, labels=labels, sources=np.asarray(sources)
            )
            print(f"  cached {cache_path.name}")

        dataset[split] = {"images": images, "labels": labels, "sources": sources}

    return dataset


def drop_cross_split_duplicates(dataset: dict[str, dict]) -> dict[str, dict]:
    """Remove training frames whose source image also appears in valid or test.

    The Roboflow split was made *after* augmentation, so a handful of source
    frames have exposure-jittered siblings on both sides of the split boundary.
    Leaving them in would let the model memorise near-identical pixels and
    inflate the held-out scores.  Only the training split is trimmed, so the
    evaluation sets stay exactly as published.
    """
    held_out = set(dataset["valid"]["sources"]) | set(dataset["test"]["sources"])
    keep = np.array([source not in held_out for source in dataset["train"]["sources"]])
    removed = int((~keep).sum())

    if removed:
        print(f"  removed {removed} training frames that duplicate valid/test sources")
        dataset["train"] = {
            "images": dataset["train"]["images"][keep],
            "labels": dataset["train"]["labels"][keep],
            "sources": [s for s, k in zip(dataset["train"]["sources"], keep) if k],
        }

    return dataset


def channel_statistics(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation over a ``(n, C, H, W)`` stack.

    Computed in float64 over the training split only -- using the evaluation
    splits here would leak their distribution into the model.  The accumulation
    is a two-pass (mean, then deviation) scheme rather than the algebraically
    equivalent ``E[x^2] - E[x]^2``, which cancels catastrophically when the mean
    is large relative to the spread, as it is for 8-bit pixel data.
    """
    stack = images.astype(np.float64)
    mean = stack.mean(axis=(0, 2, 3), keepdims=True)
    std = np.sqrt(((stack - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True))
    return mean.astype(DTYPE), np.maximum(std, 1e-6).astype(DTYPE)


def standardise(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Map raw uint8 frames to zero-mean, unit-variance float32 tensors.

    Standardisation is what makes the He initialisation below valid: the
    variance-preservation argument assumes unit-variance inputs, and without it
    the first layer's pre-activations would be an order of magnitude too large
    and would saturate the loss.
    """
    return (images.astype(DTYPE) - mean) / std


# ======================================================================
#  3.  Numerically stable scalar primitives
# ======================================================================
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function evaluated without overflow.

    ``exp(-z)`` overflows for z < -88 in float32.  Evaluating the branch
    ``exp(z) / (1 + exp(z))`` for negative ``z`` and ``1 / (1 + exp(-z))`` for
    positive ``z`` keeps every exponential argument non-positive, so the worst
    case underflows to zero instead of overflowing to infinity.
    """
    out = np.empty_like(z, dtype=DTYPE)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def bce_with_logits(
    logits: np.ndarray, targets: np.ndarray, negative_weight: float
) -> tuple[float, np.ndarray]:
    """Class-weighted binary cross-entropy evaluated from logits.

    Taking the sigmoid first and then its logarithm loses all precision once the
    sigmoid saturates (``log(0) = -inf``).  Folding the two together gives the
    log-sum-exp form

        L(z, y) = max(z, 0) - z*y + log(1 + exp(-|z|)),

    in which the exponential argument is never positive, so the expression is
    exact over the whole float32 range.  ``log1p`` is used for the final term
    because ``log(1 + x)`` loses precision for small ``x``.

    The positive class carries ~80 % of this dataset, so the negative class is
    up-weighted by ``negative_weight`` so that the two classes contribute
    equally to the gradient. (Note this is the opposite convention to the
    ``pos_weight`` argument of common framework implementations, which
    up-weights the positive class instead.)

    Parameters
    ----------
    logits : ndarray, shape (n,)
        Pre-sigmoid network output ``z``.
    targets : ndarray, shape (n,)
        Ground-truth labels in {0, 1}.
    negative_weight : float
        Multiplier applied to the loss of negative (non-defective) samples.

    Returns
    -------
    loss : float
        Weighted mean loss over the batch.
    dlogits : ndarray, shape (n,)
        Gradient of ``loss`` with respect to ``logits``.
    """
    logits = logits.astype(np.float64)
    targets = targets.astype(np.float64)

    weights = np.where(targets > 0.5, 1.0, negative_weight)
    per_sample = (
        np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    )
    loss = float((weights * per_sample).sum() / weights.sum())

    # d/dz [ max(z,0) - z*y + log(1 + exp(-|z|)) ] = sigmoid(z) - y
    dlogits = weights * (sigmoid(logits.astype(DTYPE)).astype(np.float64) - targets)
    dlogits /= weights.sum()

    return loss, dlogits.astype(DTYPE)


# ======================================================================
#  4.  Layers
# ======================================================================
# Every layer exposes the same three-part contract:
#   forward(x)      -> activations, caching whatever backward() needs
#   backward(dy)    -> gradient w.r.t. its input, filling self.grads
#   params / grads  -> dicts of trainable arrays, mutated in place by Adam
# Tensors are stored in NCHW order (batch, channel, row, column).


def he_normal(
    shape: tuple[int, ...], fan_in: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw weights from N(0, 2 / fan_in) -- the He/Kaiming initialisation.

    For a linear map ``z = Wx`` with independent zero-mean entries,
    ``Var[z] = fan_in * Var[W] * Var[x]``.  A ReLU zeroes half the distribution
    and therefore halves the variance, so choosing ``Var[W] = 2 / fan_in``
    makes activation variance neither grow nor decay with depth.  Getting this
    wrong by a constant factor per layer compounds geometrically and is the
    usual cause of a network that will not start learning.
    """
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape).astype(DTYPE)


class Conv2D:
    """Same-padded, unit-stride 2-D convolution implemented as a single GEMM.

    The naive six-nested-loop convolution is unusably slow in Python.  Instead
    each sliding window is materialised as a column of a matrix (the ``im2col``
    transform) so the whole layer becomes one dense matrix product, which NumPy
    dispatches to a tuned BLAS kernel.  The cost is memory: the column matrix is
    ``k^2`` times larger than the input.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: int, rng):
        fan_in = in_channels * kernel * kernel
        self.kernel = kernel
        self.pad = kernel // 2  # "Same" padding: output grid matches input grid.
        self.params = {
            "W": he_normal((out_channels, fan_in), fan_in, rng),
            "b": np.zeros(out_channels, dtype=DTYPE),
        }
        self.grads = {"W": None, "b": None}
        self._cache = None

    def _im2col(self, padded: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """Gather every ``k x k`` receptive field into ``(n, C*k*k, out_h*out_w)``.

        The gather loops over the ``k*k`` kernel offsets (9 iterations here) and
        slices the whole batch at once, rather than looping over the millions of
        output pixels.
        """
        n, channels, _, _ = padded.shape
        k = self.kernel
        columns = np.empty((n, channels, k, k, out_h, out_w), dtype=DTYPE)
        for row in range(k):
            for col in range(k):
                columns[:, :, row, col] = padded[
                    :, :, row : row + out_h, col : col + out_w
                ]
        return columns.reshape(n, channels * k * k, out_h * out_w)

    def _col2im(self, columns: np.ndarray, padded_shape: tuple, out_h: int, out_w: int):
        """Adjoint of ``_im2col``: scatter-add columns back to image layout.

        A pixel inside the image is read by ``k*k`` different windows, so the
        adjoint must *accumulate* rather than assign -- this is exactly the sum
        over window positions in the chain rule.
        """
        n, channels, _, _ = padded_shape
        k = self.kernel
        columns = columns.reshape(n, channels, k, k, out_h, out_w)
        padded = np.zeros(padded_shape, dtype=DTYPE)
        for row in range(k):
            for col in range(k):
                padded[:, :, row : row + out_h, col : col + out_w] += columns[
                    :, :, row, col
                ]
        return padded

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, _, height, width = x.shape
        pad = self.pad
        padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))  # zero padding
        columns = self._im2col(padded, height, width)

        # (out_channels, fan_in) @ (n, fan_in, HW) -> (n, out_channels, HW)
        out = np.einsum("of,nfp->nop", self.params["W"], columns, optimize=True)
        out += self.params["b"][None, :, None]

        self._cache = (columns, padded.shape, height, width)
        return out.reshape(n, -1, height, width)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        columns, padded_shape, height, width = self._cache
        n = dout.shape[0]
        dout = dout.reshape(n, -1, height * width)

        # dL/dW = sum over batch and pixels of (upstream gradient) x (patch)
        self.grads["W"] = np.einsum("nop,nfp->of", dout, columns, optimize=True)
        self.grads["b"] = dout.sum(axis=(0, 2))

        dcolumns = np.einsum("of,nop->nfp", self.params["W"], dout, optimize=True)
        dpadded = self._col2im(dcolumns, padded_shape, height, width)

        pad = self.pad
        return dpadded[:, :, pad : pad + height, pad : pad + width]


class ReLU:
    """Rectifier ``max(0, x)``.

    Not differentiable at the origin; the subgradient 0 is used there.  The
    kink is what makes the network non-linear, and its exactly-zero gradient on
    the negative side is also what makes the finite-difference gradient check
    below fail if a probe lands too close to a kink, hence the offset used
    there.
    """

    def __init__(self):
        self.params, self.grads = {}, {}
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask


class MaxPool2D:
    """Non-overlapping 2x2 max pooling.

    Halves each spatial dimension, which quarters the work of every subsequent
    layer and doubles the receptive field in input pixels.  Max rather than mean
    pooling is used because a defect is a *local* anomaly: a wrinkle occupying
    one cell of the window should survive pooling, not be averaged away.

    An odd input dimension is handled by cropping the trailing row/column,
    which is simpler and less arbitrary than padding with -inf.
    """

    def __init__(self, size: int = 2):
        self.size = size
        self.params, self.grads = {}, {}
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        s = self.size
        n, channels, height, width = x.shape
        height, width = (height // s) * s, (width // s) * s
        cropped = x[:, :, :height, :width]

        windows = cropped.reshape(n, channels, height // s, s, width // s, s)
        # Flatten the two window axes so a single argmax identifies the winner.
        flat = windows.transpose(0, 1, 2, 4, 3, 5).reshape(
            n, channels, height // s, width // s, s * s
        )
        winners = flat.argmax(axis=-1)

        self._cache = (x.shape, winners, height, width)
        return np.take_along_axis(flat, winners[..., None], axis=-1).squeeze(-1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        s = self.size
        input_shape, winners, height, width = self._cache
        n, channels = input_shape[0], input_shape[1]

        # Route the whole upstream gradient to the argmax cell; all other cells
        # of the window had no influence on the output and receive zero.
        flat = np.zeros((n, channels, height // s, width // s, s * s), dtype=DTYPE)
        np.put_along_axis(flat, winners[..., None], dout[..., None], axis=-1)

        windows = flat.reshape(n, channels, height // s, width // s, s, s)
        cropped = windows.transpose(0, 1, 2, 4, 3, 5).reshape(
            n, channels, height, width
        )

        dx = np.zeros(input_shape, dtype=DTYPE)
        dx[:, :, :height, :width] = cropped  # Cropped rows/columns get zero gradient.
        return dx


class Flatten:
    """Reshape ``(n, C, H, W)`` to ``(n, C*H*W)``; no parameters, no arithmetic."""

    def __init__(self):
        self.params, self.grads = {}, {}
        self._shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._shape)


class Dense:
    """Fully connected layer ``y = xW + b``."""

    def __init__(self, in_features: int, out_features: int, rng):
        self.params = {
            "W": he_normal((in_features, out_features), in_features, rng),
            "b": np.zeros(out_features, dtype=DTYPE),
        }
        self.grads = {"W": None, "b": None}
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.grads["W"] = self._x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T


# ======================================================================
#  5.  Network
# ======================================================================
class DefectNet:
    """Stack of layers mapping a standardised frame to a single defect logit."""

    def __init__(self, rng: np.random.Generator):
        self.layers: list = []
        channels, height, width = IMG_C, IMG_H, IMG_W

        for out_channels in CONV_CHANNELS:
            self.layers += [
                Conv2D(channels, out_channels, KERNEL, rng),
                ReLU(),
                MaxPool2D(2),
            ]
            channels, height, width = out_channels, height // 2, width // 2

        self.layers += [
            Flatten(),
            Dense(channels * height * width, DENSE_UNITS, rng),
            ReLU(),
            Dense(DENSE_UNITS, 1, rng),
        ]

    # -- inference -----------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the logit (not the probability) for each image in ``x``."""
        for layer in self.layers:
            x = layer.forward(x)
        return x.ravel()

    def backward(self, dlogits: np.ndarray) -> None:
        """Propagate ``dL/dlogits`` back through every layer, filling ``grads``."""
        grad = dlogits.reshape(-1, 1)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict_logits(self, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Logits for a whole split, evaluated in batches to bound peak memory.

        The im2col buffers scale with the batch size, so scoring 3000 images in
        one call would need gigabytes; 64 at a time keeps the working set small
        while still giving BLAS a large enough matrix to be efficient.
        """
        chunks = [
            self.forward(x[i : i + batch_size]) for i in range(0, len(x), batch_size)
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=DTYPE)

    def predict_proba(self, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Defect probability in [0, 1] for each image."""
        return sigmoid(self.predict_logits(x, batch_size))

    # -- parameter access ----------------------------------------------
    def parameter_items(self):
        """Yield ``(layer_index, name)`` for every trainable array."""
        for index, layer in enumerate(self.layers):
            for name in layer.params:
                yield index, name

    def parameter_count(self) -> int:
        return sum(self.layers[i].params[n].size for i, n in self.parameter_items())

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            f"{i}.{n}": self.layers[i].params[n].copy()
            for i, n in self.parameter_items()
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for key, value in state.items():
            index, name = key.split(".")
            self.layers[int(index)].params[name] = np.asarray(value, dtype=DTYPE).copy()

    def flops_per_image(self) -> int:
        """Multiply-accumulate count (x2 for the add) of one forward pass.

        Reported so the runtime measured in the optimisation study can be
        compared against the arithmetic the method actually requires.
        """
        total, height, width = 0, IMG_H, IMG_W
        channels = IMG_C
        for out_channels in CONV_CHANNELS:
            total += 2 * height * width * out_channels * channels * KERNEL * KERNEL
            channels, height, width = out_channels, height // 2, width // 2
        flat = channels * height * width
        total += 2 * flat * DENSE_UNITS + 2 * DENSE_UNITS
        return total


# ======================================================================
#  6.  Optimiser
# ======================================================================
class Adam:
    """Adam with bias correction and decoupled L2 weight decay.

    Adam rescales each parameter's step by an estimate of the root-mean-square
    of its recent gradients,

        m_t = b1*m_{t-1} + (1-b1)*g,      v_t = b2*v_{t-1} + (1-b2)*g^2,
        step = lr * m_hat / (sqrt(v_hat) + eps),

    which makes a single learning rate workable across layers whose gradient
    magnitudes differ by orders of magnitude -- here the convolution kernels and
    the 2240-input dense layer.  ``m`` and ``v`` start at zero and are therefore
    biased towards zero for the first few steps; dividing by ``1 - b^t``
    (``m_hat``, ``v_hat``) removes that bias exactly.

    Weight decay is applied to the parameter directly rather than folded into
    the gradient, so the decay rate is not rescaled by ``v_hat`` (AdamW).
    Biases are excluded, since shrinking them only shifts the decision boundary.
    """

    def __init__(self, model: DefectNet, lr: float, weight_decay: float):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_count = 0
        self.moment1 = {key: np.zeros_like(self._param(key)) for key in self._keys()}
        self.moment2 = {key: np.zeros_like(self._param(key)) for key in self._keys()}

    def _keys(self):
        return list(self.model.parameter_items())

    def _param(self, key):
        index, name = key
        return self.model.layers[index].params[name]

    def step(self) -> None:
        self.step_count += 1
        bias1 = 1.0 - ADAM_BETA1**self.step_count
        bias2 = 1.0 - ADAM_BETA2**self.step_count

        for key in self._keys():
            index, name = key
            layer = self.model.layers[index]
            grad = layer.grads[name]
            if grad is None:
                continue

            self.moment1[key] = ADAM_BETA1 * self.moment1[key] + (1 - ADAM_BETA1) * grad
            self.moment2[key] = (
                ADAM_BETA2 * self.moment2[key] + (1 - ADAM_BETA2) * grad * grad
            )

            m_hat = self.moment1[key] / bias1
            v_hat = self.moment2[key] / bias2

            if name == "W" and self.weight_decay:
                layer.params[name] -= self.lr * self.weight_decay * layer.params[name]
            layer.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)


# ======================================================================
#  7.  Verification of the analytic gradient
# ======================================================================
@contextmanager
def high_precision():
    """Run the network in float64 for the duration of the block.

    Training deliberately uses float32: it halves memory traffic and BLAS is
    roughly twice as fast on it.  A finite-difference check cannot tolerate that
    precision, though -- see ``gradient_check`` -- so the check temporarily
    promotes the module-wide working precision.
    """
    global DTYPE
    saved = DTYPE
    DTYPE = np.float64
    try:
        yield
    finally:
        DTYPE = saved


def gradient_check(
    rng: np.random.Generator,
    n_probes: int = 25,
    step_sizes: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
) -> dict[float, float]:
    """Verify backpropagation against central differences over a sweep of steps.

    Backpropagation here is a hand-derived chain rule across ten layers, and a
    transposed index or a missing accumulation produces a network that still
    trains -- just worse -- so the error is easy to miss without an independent
    check.  That check is the central difference

        dL/dw ~ [L(w + h) - L(w - h)] / (2h).

    Choosing ``h`` is the whole difficulty.  The total error is the sum of two
    competing terms:

    * **truncation**, from the neglected Taylor remainder, of order
      ``h^2 * L'''(w) / 6`` -- it shrinks as ``h`` shrinks;
    * **round-off**, from cancellation in the numerator, of order
      ``eps * |L| / h`` -- it *grows* as ``h`` shrinks.

    Their sum is minimised near ``h ~ eps^(1/3)``, which is about 6e-6 in
    float64 but about 5e-3 in float32.  At that float32 optimum the truncation
    term is already large, so the check is run in float64 instead: the sweep
    below reproduces the classic V-shaped error curve and confirms the analytic
    gradient to about 1e-10 at the optimum, a margin no float32 check could
    resolve.

    Two further details matter for a ReLU network.  A large ``h`` can push a
    pre-activation across the kink at zero, where the loss is not differentiable
    and the two-sided difference is legitimately wrong; that pollutes the
    *worst-case* error at large ``h`` but not the median, so the median over
    probes is the statistic reported.  Probes are also restricted to parameters
    whose analytic gradient is not already negligible, because the relative
    error of a near-zero quantity carries no information.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of the synthetic batch, the network weights and the probe sites.
    n_probes : int
        Number of scalar parameters perturbed.  Each costs two forward passes
        per step size.
    step_sizes : tuple of float
        Perturbations ``h`` to sweep.  The same probe sites are reused for every
        ``h`` so the resulting curve is directly comparable.

    Returns
    -------
    dict[float, float]
        Median relative error ``|analytic - numeric| / (|analytic| + |numeric|)``
        for each step size.  A minimum below about 1e-6 verifies the gradients.
    """
    with high_precision():
        model = DefectNet(rng)
        # A synthetic batch suffices: correctness of the chain rule does not
        # depend on the data, and four images keep each probe cheap.
        x = rng.normal(size=(4, IMG_C, IMG_H, IMG_W)).astype(DTYPE)
        y = np.array([0, 1, 1, 0], dtype=DTYPE)

        def loss() -> float:
            value, _ = bce_with_logits(model.forward(x), y, negative_weight=1.0)
            return value

        # One analytic pass populates every layer's grads.
        _, dlogits = bce_with_logits(model.forward(x), y, negative_weight=1.0)
        model.backward(dlogits)
        analytic = {
            key: model.layers[key[0]].grads[key[1]].copy()
            for key in model.parameter_items()
        }

        keys = list(model.parameter_items())
        probes = []
        while len(probes) < n_probes:
            index, name = keys[rng.integers(len(keys))]
            cell = tuple(
                rng.integers(dim) for dim in model.layers[index].params[name].shape
            )
            if abs(float(analytic[(index, name)][cell])) > 1e-8:
                probes.append((index, name, cell))

        curve = {}
        for h in step_sizes:
            errors = []
            for index, name, cell in probes:
                param = model.layers[index].params[name]
                original = float(param[cell])

                param[cell] = original + h
                loss_plus = loss()
                param[cell] = original - h
                loss_minus = loss()
                param[cell] = original  # restore before moving to the next probe

                numeric = (loss_plus - loss_minus) / (2.0 * h)
                exact = float(analytic[(index, name)][cell])
                errors.append(
                    abs(exact - numeric) / max(abs(exact) + abs(numeric), 1e-30)
                )
            curve[h] = float(np.median(errors))

    return curve


# ======================================================================
#  8.  Evaluation metrics
# ======================================================================
@dataclass
class Metrics:
    """Confusion-matrix-derived scores for one split at one threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    auc: float

    def summary(self) -> str:
        return (
            f"    threshold          {self.threshold:.3f}\n"
            f"    confusion [TP FP]  [{self.tp:4d} {self.fp:4d}]\n"
            f"              [FN TN]  [{self.fn:4d} {self.tn:4d}]\n"
            f"    accuracy           {self.accuracy:.3f}\n"
            f"    balanced accuracy  {self.balanced_accuracy:.3f}\n"
            f"    precision          {self.precision:.3f}\n"
            f"    recall (defect)    {self.recall:.3f}\n"
            f"    specificity (ok)   {self.specificity:.3f}\n"
            f"    F1                 {self.f1:.3f}\n"
            f"    ROC AUC            {self.auc:.3f}"
        )


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sweep every attainable threshold and return ``(fpr, tpr)``.

    Sorting the scores once and taking cumulative sums gives the full curve in
    O(n log n) instead of re-scoring the set at each of n candidate thresholds.
    """
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]

    tps = np.cumsum(ranked == 1)
    fps = np.cumsum(ranked == 0)
    n_pos, n_neg = max(int((labels == 1).sum()), 1), max(int((labels == 0).sum()), 1)

    tpr = np.concatenate(([0.0], tps / n_pos))
    fpr = np.concatenate(([0.0], fps / n_neg))
    return fpr, tpr


def trapezoid_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the ROC curve by the composite trapezoidal rule.

    The ROC is a piecewise-linear staircase through a finite set of points, so
    the trapezoidal rule is not an approximation here -- it integrates the
    curve exactly.  A higher-order rule (Simpson) would be strictly worse,
    because it assumes a smoothness the staircase does not have.
    """
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Metrics:
    """Score a set of predicted probabilities against the ground truth."""
    predicted = scores >= threshold
    actual = labels == 1

    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    tn = int(np.sum(~predicted & ~actual))
    fn = int(np.sum(~predicted & actual))

    def safe_ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    precision = safe_ratio(tp, tp + fp)

    fpr, tpr = roc_curve(scores, labels)
    return Metrics(
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=safe_ratio(tp + tn, tp + tn + fp + fn),
        balanced_accuracy=0.5 * (recall + specificity),
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=safe_ratio(2 * precision * recall, precision + recall)
        if precision + recall
        else 0.0,
        auc=trapezoid_auc(fpr, tpr),
    )


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pick the decision threshold that maximises Youden's J on validation.

    With ~85 % of frames defective, the default 0.5 cut and plain accuracy both
    reward a model that simply always says "defect".  Youden's J
    (``sensitivity + specificity - 1``) weights the two classes equally, so the
    chosen threshold reflects detection quality rather than the class prior.
    The threshold is fitted on *validation* data only and then frozen, so the
    test score remains an honest estimate.
    """
    candidates = np.unique(np.round(scores, 4))
    best_threshold, best_j = 0.5, -np.inf
    for threshold in candidates:
        metrics = evaluate(scores, labels, float(threshold))
        j = metrics.recall + metrics.specificity - 1.0
        if j > best_j:
            best_threshold, best_j = float(threshold), j
    return best_threshold


# ======================================================================
#  9.  Training
# ======================================================================
def train(
    model: DefectNet,
    data: dict[str, dict],
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """Mini-batch training loop with validation-based model selection.

    Returns the per-epoch history and leaves ``model`` holding the weights of
    the epoch with the best validation balanced accuracy (not the last epoch --
    on a set this small the later epochs overfit).
    """
    x_train = standardise(data["train"]["images"], mean, std)
    y_train = data["train"]["labels"].astype(DTYPE)
    x_valid = standardise(data["valid"]["images"], mean, std)
    y_valid = data["valid"]["labels"].astype(DTYPE)

    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    negative_weight = n_pos / max(n_neg, 1.0)  # equalises the two classes' total weight
    print(
        f"  class balance: {int(n_pos)} defective / {int(n_neg)} clean "
        f"-> negative-class weight {negative_weight:.2f}"
    )

    optimiser = Adam(model, lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "valid_loss": [], "train_bacc": [], "valid_bacc": []}
    best_state, best_bacc, best_epoch = model.state_dict(), -np.inf, 0

    n = len(x_train)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        order = rng.permutation(n)  # reshuffled each epoch to decorrelate batches
        running_loss, seen = 0.0, 0

        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            logits = model.forward(x_train[batch])
            loss, dlogits = bce_with_logits(logits, y_train[batch], negative_weight)
            model.backward(dlogits)
            optimiser.step()

            running_loss += loss * len(batch)
            seen += len(batch)

        train_scores = model.predict_proba(x_train)
        valid_logits = model.predict_logits(x_valid)
        valid_scores = sigmoid(valid_logits)
        valid_loss, _ = bce_with_logits(valid_logits, y_valid, negative_weight)
        train_bacc = evaluate(
            train_scores, data["train"]["labels"], 0.5
        ).balanced_accuracy
        valid_bacc = evaluate(
            valid_scores, data["valid"]["labels"], 0.5
        ).balanced_accuracy

        history["train_loss"].append(running_loss / seen)
        history["valid_loss"].append(valid_loss)
        history["train_bacc"].append(train_bacc)
        history["valid_bacc"].append(valid_bacc)

        if valid_bacc > best_bacc:
            best_state, best_bacc, best_epoch = model.state_dict(), valid_bacc, epoch

        print(
            f"  epoch {epoch:3d}/{epochs}  "
            f"train loss {running_loss / seen:.4f}  valid loss {valid_loss:.4f}  "
            f"train bacc {train_bacc:.3f}  valid bacc {valid_bacc:.3f}  "
            f"({time.perf_counter() - started:.1f} s)"
        )

    print(
        f"  restoring epoch {best_epoch} (validation balanced accuracy {best_bacc:.3f})"
    )
    model.load_state_dict(best_state)
    return history


# ======================================================================
#  10.  Figures
# ======================================================================
def plot_history(history: dict[str, list[float]]) -> None:
    """Loss and balanced accuracy against epoch, for the report."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 0.42))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["valid_loss"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted BCE loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_bacc"], label="train")
    axes[1].plot(epochs, history["valid_bacc"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "training-history.pdf")
    plt.close(figure)


def plot_gradient_check(curve: dict[float, float]) -> None:
    """Finite-difference error against step size, on log-log axes.

    The V shape is the signature of the two competing error terms: the falling
    left branch is truncation (slope +2 in h), the rising right branch is
    round-off (slope -1 in h).
    """
    steps = np.array(sorted(curve))
    errors = np.array([curve[h] for h in steps])

    figure_, ax = plt.subplots()
    ax.loglog(steps, np.maximum(errors, 1e-16), "o-", label="measured")
    best = steps[np.argmin(errors)]
    exponent = int(round(np.log10(best)))
    ax.axvline(best, ls="--", color="grey", label=f"optimum $h = 10^{{{exponent}}}$")
    ax.set_xlabel("Perturbation $h$")
    ax.set_ylabel("Median relative error")
    ax.legend()

    figure_.tight_layout()
    figure_.savefig(FIGURE_DIR / "gradient-check.pdf")
    plt.close(figure_)


def plot_roc(scores: np.ndarray, labels: np.ndarray, auc: float) -> None:
    """ROC curve with the chance diagonal for reference."""
    fpr, tpr = roc_curve(scores, labels)

    figure, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"DefectNet (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "roc-test.pdf")
    plt.close(figure)


def plot_confusion(metrics: Metrics) -> None:
    """Confusion matrix as an annotated heat map."""
    # Rows are predictions, columns the ground truth, matching the printed
    # table: [[TP, FP], [FN, TN]].
    matrix = np.array([[metrics.tp, metrics.fp], [metrics.fn, metrics.tn]])

    figure, ax = plt.subplots(figsize=(FIG_WIDTH * 0.6, FIG_WIDTH * 0.6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["defect", "clean"])
    ax.set_yticks([0, 1], ["defect", "clean"])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    limit = matrix.max()
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > limit / 2 else "black",
            )

    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "confusion-test.pdf")
    plt.close(figure)


# ======================================================================
#  11.  Entry point
# ======================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_L2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--rebuild-cache", action="store_true", help="re-decode the JPEGs"
    )
    parser.add_argument("--skip-gradcheck", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    # One seeded generator drives initialisation, batch shuffling and the
    # gradient probes, so a given seed reproduces the run exactly.
    rng = np.random.default_rng(args.seed)
    FIGURE_DIR.mkdir(exist_ok=True)
    ARTEFACT_DIR.mkdir(exist_ok=True)

    print("[1/5] Loading data")
    data = drop_cross_split_duplicates(load_dataset(rebuild=args.rebuild_cache))
    for split in SPLITS:
        labels = data[split]["labels"]
        print(
            f"  {split:5s}: {len(labels):5d} images, {int(labels.sum()):5d} defective "
            f"({labels.mean() * 100:.1f} %)"
        )

    mean, std = channel_statistics(data["train"]["images"])
    print(f"  train channel means  (R,G,B): {np.round(mean.ravel(), 2)}")
    print(f"  train channel stddev (R,G,B): {np.round(std.ravel(), 2)}")

    print("\n[2/5] Building network")
    model = DefectNet(rng)
    print(f"  input  {IMG_C}x{IMG_H}x{IMG_W} standardised float32")
    print(f"  layers {' -> '.join(type(layer).__name__ for layer in model.layers)}")
    print(f"  trainable parameters: {model.parameter_count():,}")
    print(f"  forward cost: {model.flops_per_image() / 1e6:.2f} MFLOP per image")

    gradient_curve = None
    if not args.skip_gradcheck:
        print("\n[3/5] Verifying backpropagation against central differences")
        started = time.perf_counter()
        gradient_curve = gradient_check(np.random.default_rng(args.seed + 1))
        for step, error in sorted(gradient_curve.items(), reverse=True):
            print(f"    h = {step:.0e}   median relative error {error:.2e}")
        best_error = min(gradient_curve.values())
        verdict = "PASS" if best_error < 1e-6 else "FAIL"
        print(
            f"  best {best_error:.3e}  [{verdict}]  ({time.perf_counter() - started:.1f} s)"
        )
    else:
        print("\n[3/5] Gradient check skipped")

    print("\n[4/5] Training")
    started = time.perf_counter()
    history = train(
        model,
        data,
        mean,
        std,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rng=rng,
    )
    train_seconds = time.perf_counter() - started
    print(
        f"  total training time {train_seconds:.1f} s "
        f"({train_seconds / max(args.epochs, 1):.1f} s per epoch)"
    )

    print("\n[5/5] Evaluating")
    valid_scores = model.predict_proba(standardise(data["valid"]["images"], mean, std))
    threshold = choose_threshold(valid_scores, data["valid"]["labels"])

    started = time.perf_counter()
    test_x = standardise(data["test"]["images"], mean, std)
    test_scores = model.predict_proba(test_x)
    inference_ms = 1e3 * (time.perf_counter() - started) / len(test_x)

    valid_metrics = evaluate(valid_scores, data["valid"]["labels"], threshold)
    test_metrics = evaluate(test_scores, data["test"]["labels"], threshold)
    default_metrics = evaluate(test_scores, data["test"]["labels"], 0.5)

    # Reference point: the trivial classifier that calls every frame defective.
    # Anything that cannot beat this on balanced accuracy has learned nothing.
    majority = evaluate(np.ones_like(test_scores), data["test"]["labels"], 0.5)

    print("\n  validation (threshold fitted here)")
    print(valid_metrics.summary())
    print("\n  test, tuned threshold")
    print(test_metrics.summary())
    print("\n  test, default 0.5 threshold")
    print(default_metrics.summary())
    print("\n  test, always-defect baseline")
    print(majority.summary())
    print(f"\n  inference {inference_ms:.2f} ms per image")

    np.savez_compressed(
        ARTEFACT_DIR / "model.npz",
        threshold=np.float32(threshold),
        mean=mean,
        std=std,
        **model.state_dict(),
    )
    with (ARTEFACT_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": vars(args),
                "parameters": model.parameter_count(),
                "flops_per_image": model.flops_per_image(),
                "train_seconds": train_seconds,
                "inference_ms_per_image": inference_ms,
                "gradient_check": gradient_curve,
                "history": history,
                "validation": vars(valid_metrics),
                "test_tuned": vars(test_metrics),
                "test_default": vars(default_metrics),
                "test_majority_baseline": vars(majority),
            },
            handle,
            indent=2,
        )

    if not args.no_figures:
        if gradient_curve is not None:
            plot_gradient_check(gradient_curve)
        plot_history(history)
        plot_roc(test_scores, data["test"]["labels"], test_metrics.auc)
        plot_confusion(test_metrics)
        print(f"  figures written to {FIGURE_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"  model and metrics written to {ARTEFACT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
