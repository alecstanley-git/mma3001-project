"""Turning photographs into numbers a regression network can learn from.

The network in :mod:`packaging_defects.network` is the kind taught in the unit:
it takes a list of numerical *features* and regresses numerical outputs.  A raw
photograph is hundreds of thousands of pixel values, most of which carry little
information on their own, so each photograph is first summarised.

The photograph is cut into small square **blocks** (10 by 10 reduced pixels).
For each block this module records:

* **Edge directions** - how strongly the brightness changes in each of nine
  directions.  Brightness change is estimated with *central finite
  differences*, the derivative approximation taught in the unit.  Torn film,
  wrinkles and loose meat all show up as edges in unusual directions.
* **Edge strength** - the total amount of brightness change in the block.
* **Average colour** - the mean red, green and blue values.
* **Brightness spread** - the standard deviation of brightness.

The result is a small "feature map": a grid of blocks with fourteen numbers
per block.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import load_images

#: Side length of a block, in reduced pixels.
BLOCK_SIZE = 10

#: Number of edge directions recorded per block (each covers 20 degrees).
NUMBER_OF_DIRECTIONS = 9

#: Numbers recorded per block: 9 edge directions, edge strength, 3 colours,
#: brightness spread.
FEATURES_PER_BLOCK = NUMBER_OF_DIRECTIONS + 5

# Standard weights for converting red, green and blue to perceived brightness.
BRIGHTNESS_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def brightness_gradients(brightness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate how quickly brightness changes across and down each image.

    Uses the central finite difference

    $$\\frac{\\partial f}{\\partial x} \\approx \\frac{f(x+1) - f(x-1)}{2}$$

    whose error shrinks with the square of the pixel spacing.  At the image
    border, where one neighbour is missing, the one-sided difference
    ``f(x+1) - f(x)`` is used instead.

    Parameters
    ----------
    brightness : numpy.ndarray, shape (n, height, width)
        Brightness of each pixel.

    Returns
    -------
    across, down : numpy.ndarray, shape (n, height, width)
        Rate of change in the horizontal and vertical directions.
    """
    across = np.empty_like(brightness)
    across[:, :, 1:-1] = (brightness[:, :, 2:] - brightness[:, :, :-2]) / 2.0
    across[:, :, 0] = brightness[:, :, 1] - brightness[:, :, 0]
    across[:, :, -1] = brightness[:, :, -1] - brightness[:, :, -2]

    down = np.empty_like(brightness)
    down[:, 1:-1, :] = (brightness[:, 2:, :] - brightness[:, :-2, :]) / 2.0
    down[:, 0, :] = brightness[:, 1, :] - brightness[:, 0, :]
    down[:, -1, :] = brightness[:, -1, :] - brightness[:, -2, :]
    return across, down


def sum_over_blocks(values: np.ndarray, block_size: int) -> np.ndarray:
    """Add up pixel values inside each block.

    Parameters
    ----------
    values : numpy.ndarray, shape (n, height, width)
        One value per pixel.
    block_size : int
        Side length of a block; must divide the height and width.

    Returns
    -------
    numpy.ndarray, shape (n, height / block_size, width / block_size)
    """
    count, height, width = values.shape
    blocks = values.reshape(count, height // block_size, block_size, width // block_size, block_size)
    return blocks.sum(axis=(2, 4))


def block_features(images: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Summarise each photograph as a grid of blocks.

    Parameters
    ----------
    images : numpy.ndarray, shape (n, height, width, 3), dtype uint8
        Reduced photographs.  Height and width must be multiples of
        ``block_size``.
    block_size : int
        Side length of a block in pixels.

    Returns
    -------
    numpy.ndarray, shape (n, height / block_size, width / block_size, 14)
        The fourteen numbers described in the module documentation, for every
        block of every photograph.

    Raises
    ------
    ValueError
        If the image size is not a multiple of ``block_size``.
    """
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must have shape (n, height, width, 3)")
    count, height, width, _ = images.shape
    if height % block_size or width % block_size:
        raise ValueError(f"image size {width}x{height} is not a multiple of {block_size}")

    pixels = images.astype(np.float32) / 255.0
    brightness = pixels @ BRIGHTNESS_WEIGHTS
    across, down = brightness_gradients(brightness)
    strength = np.hypot(across, down)

    # An edge and its mirror image point in opposite directions but look the
    # same, so directions are folded into the half circle 0 to 180 degrees.
    angle = np.mod(np.arctan2(down, across), np.pi)
    position = angle / (np.pi / NUMBER_OF_DIRECTIONS)
    lower = np.floor(position).astype(np.int64) % NUMBER_OF_DIRECTIONS
    upper = (lower + 1) % NUMBER_OF_DIRECTIONS
    # An edge that falls between two directions is shared between them in
    # proportion to how close it is to each, which avoids sudden jumps.
    share_of_upper = position - np.floor(position)

    rows, columns = height // block_size, width // block_size
    features = np.empty((count, rows, columns, FEATURES_PER_BLOCK), dtype=np.float32)
    for direction in range(NUMBER_OF_DIRECTIONS):
        contribution = strength * (
            (lower == direction) * (1.0 - share_of_upper) + (upper == direction) * share_of_upper
        )
        features[..., direction] = sum_over_blocks(contribution, block_size)

    # Divide by the block's total so the directions describe the *pattern* of
    # edges regardless of lighting; the total itself is kept separately.
    directions = features[..., :NUMBER_OF_DIRECTIONS]
    total = np.sqrt((directions**2).sum(axis=-1, keepdims=True))
    features[..., :NUMBER_OF_DIRECTIONS] = directions / (total + 1e-3)
    features[..., NUMBER_OF_DIRECTIONS] = np.log1p(total[..., 0])

    pixels_per_block = block_size * block_size
    for channel in range(3):
        features[..., NUMBER_OF_DIRECTIONS + 1 + channel] = (
            sum_over_blocks(pixels[..., channel], block_size) / pixels_per_block
        )

    mean_brightness = sum_over_blocks(brightness, block_size) / pixels_per_block
    mean_square = sum_over_blocks(brightness**2, block_size) / pixels_per_block
    # Rounding can make the variance very slightly negative for flat blocks.
    features[..., -1] = np.sqrt(np.clip(mean_square - mean_brightness**2, 0.0, None))
    return features


def mirror_feature_maps(maps: np.ndarray, left_right: bool, up_down: bool) -> np.ndarray:
    """Return the block features that the mirrored photographs would have.

    Mirroring a photograph is a cheap way to make extra training examples: a
    torn seal is still a torn seal when seen in a mirror.  The features of the
    mirrored photograph do not need to be recomputed, because mirroring only
    rearranges them:

    * the blocks swap places (left with right, or top with bottom); and
    * an edge at an angle of ``a`` degrees becomes an edge at ``180 - a``
      degrees, so the nine direction values swap places too.

    Mirroring both ways at once is a half-turn, which leaves edge directions
    unchanged.

    Parameters
    ----------
    maps : numpy.ndarray, shape (n, block rows, block columns, 14)
        As returned by :func:`block_features`.
    left_right : bool
        Mirror left to right.
    up_down : bool
        Mirror top to bottom.

    Returns
    -------
    numpy.ndarray
        Same shape as ``maps``.
    """
    if left_right:
        maps = maps[:, :, ::-1]
    if up_down:
        maps = maps[:, ::-1]
    if left_right != up_down:
        order = np.arange(FEATURES_PER_BLOCK)
        order[:NUMBER_OF_DIRECTIONS] = (-np.arange(NUMBER_OF_DIRECTIONS)) % NUMBER_OF_DIRECTIONS
        maps = maps[..., order]
    return np.ascontiguousarray(maps)


def feature_maps(
    image_paths: list[Path],
    cache_file: Path | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Compute the block features of photographs stored on disk.

    Photographs are opened a few at a time so that the full-size images never
    all sit in memory together.  Because opening thousands of photographs
    takes a little while, the result can be saved to ``cache_file`` and reused
    on later runs.  A saved result is ignored if the list of files differs.

    Parameters
    ----------
    image_paths : list of pathlib.Path
        Photographs to summarise.
    cache_file : pathlib.Path, optional
        Where to save the result between runs.  ``None`` disables saving.
    batch_size : int
        Photographs opened at once.

    Returns
    -------
    numpy.ndarray, shape (n, block rows, block columns, 14)

    Raises
    ------
    OSError
        If a photograph cannot be opened.
    """
    names = np.array([Path(path).name for path in image_paths])
    if cache_file is not None and Path(cache_file).is_file():
        with np.load(cache_file) as saved:
            if np.array_equal(saved["names"], names):
                return saved["features"]

    parts = [
        block_features(load_images(image_paths[start : start + batch_size]))
        for start in range(0, len(image_paths), batch_size)
    ]
    features = np.concatenate(parts)
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, features=features, names=names)
    return features
