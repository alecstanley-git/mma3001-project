"""Tests of the block features computed from photographs."""

import numpy as np
import pytest

from packaging_defects.dataset import load_images, load_split
from packaging_defects.features import (
    FEATURES_PER_BLOCK,
    NUMBER_OF_DIRECTIONS,
    block_features,
    brightness_gradients,
    feature_maps,
    mirror_feature_maps,
    sum_over_blocks,
)


def test_gradient_of_a_ramp_is_exact():
    # Brightness rises by 3 per pixel across and falls by 2 per pixel down.
    down, across = np.mgrid[0:20, 0:30].astype(np.float32)
    ramp = (3.0 * across - 2.0 * down)[None]
    gradient_across, gradient_down = brightness_gradients(ramp)
    assert np.allclose(gradient_across, 3.0)
    assert np.allclose(gradient_down, -2.0)


def test_central_difference_error_falls_with_the_square_of_the_spacing():
    # Sample sin(x) coarsely and finely; halving the spacing should quarter the error.
    errors = []
    for points in (40, 80):
        positions = np.linspace(0.0, 2.0, points, dtype=np.float64)
        spacing = positions[1] - positions[0]
        values = np.tile(np.sin(positions), (3, 1))[None]
        slope = brightness_gradients(values)[0][0, 1, 1:-1] / spacing
        errors.append(np.abs(slope - np.cos(positions[1:-1])).max())
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.15)


def test_sum_over_blocks():
    values = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    assert sum_over_blocks(values, 2).tolist() == [[[10.0, 18.0], [42.0, 50.0]]]


def test_block_features_shape():
    images = np.zeros((2, 40, 60, 3), dtype=np.uint8)
    assert block_features(images).shape == (2, 4, 6, FEATURES_PER_BLOCK)


def test_vertical_edge_is_recorded_in_the_first_direction():
    # Dark left half, bright right half: brightness changes across, not down.
    image = np.zeros((1, 40, 40, 3), dtype=np.uint8)
    image[:, :, 20:] = 255
    features = block_features(image)
    edge_block = features[0, 1, 1]  # the block just left of the edge
    assert np.argmax(edge_block[:NUMBER_OF_DIRECTIONS]) == 0
    flat_block = features[0, 1, 0]  # a block with no edge in it
    assert np.allclose(flat_block[: NUMBER_OF_DIRECTIONS + 1], 0.0)


def test_colour_and_spread_features():
    image = np.zeros((1, 10, 10, 3), dtype=np.uint8)
    image[..., 0] = 255  # pure red, perfectly even
    block = block_features(image)[0, 0, 0]
    assert np.allclose(block[NUMBER_OF_DIRECTIONS + 1 : NUMBER_OF_DIRECTIONS + 4], [1.0, 0.0, 0.0])
    assert block[-1] == pytest.approx(0.0, abs=1e-3)


def test_wrong_image_size_is_rejected():
    with pytest.raises(ValueError):
        block_features(np.zeros((1, 45, 60, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        block_features(np.zeros((45, 60, 3), dtype=np.uint8))


@pytest.mark.parametrize("left_right, up_down", [(True, False), (False, True), (True, True)])
def test_mirrored_features_match_features_of_the_mirrored_photograph(
    tiny_dataset, left_right, up_down
):
    images = load_images(load_split(tiny_dataset, "valid").image_paths[:3])
    mirrored = images[:, ::-1] if up_down else images
    mirrored = mirrored[:, :, ::-1] if left_right else mirrored
    expected = block_features(mirrored)
    assert np.allclose(
        mirror_feature_maps(block_features(images), left_right, up_down), expected, atol=1e-4
    )


def test_saved_features_are_reused_and_refreshed(tiny_dataset, tmp_path):
    paths = load_split(tiny_dataset, "valid").image_paths[:4]
    cache_file = tmp_path / "features.npz"
    first = feature_maps(paths, cache_file)
    assert cache_file.is_file()
    assert np.array_equal(feature_maps(paths, cache_file), first)
    # A different list of photographs must not be served from the saved copy.
    assert feature_maps(paths[:2], cache_file).shape[0] == 2
