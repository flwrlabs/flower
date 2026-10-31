# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Differential Privacy (DP) utility functions tests."""


import numpy as np

from .differential_privacy import (
    CLIENTS_DISCREPANCY_WARNING,
    KEY_CLIPPING_NORM,
    KEY_NORM_BIT,
    adaptive_clip_inputs_inplace,
    add_gaussian_noise_inplace,
    clip_inputs_inplace,
    compute_adaptive_noise_params,
    compute_clip_model_update,
    compute_stdv,
    get_norm,
)


def test_differential_privacy_constants() -> None:
    """Test differential privacy constants."""
    assert KEY_CLIPPING_NORM == "clipping_norm"
    assert KEY_NORM_BIT == "norm_bit"
    assert "privacy guarantees" in CLIENTS_DISCREPANCY_WARNING


def test_add_gaussian_noise_inplace() -> None:
    """Test add_gaussian_noise_inplace function."""
    # Prepare
    update = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])]
    std_dev = 0.1

    # Execute
    add_gaussian_noise_inplace(update, std_dev)

    # Assert
    # Check that the shape of the result is the same as the input
    for layer in update:
        assert layer.shape == (2, 2)

    # Check that the values have been changed and are not equal to the original update
    for layer in update:
        assert not np.array_equal(
            layer, [[1.0, 2.0], [3.0, 4.0]]
        ) and not np.array_equal(layer, [[5.0, 6.0], [7.0, 8.0]])

    # Check that the noise has been added
    for layer in update:
        noise_added = (
            layer - np.array([[1.0, 2.0], [3.0, 4.0]])
            if np.array_equal(layer, [[1.0, 2.0], [3.0, 4.0]])
            else layer - np.array([[5.0, 6.0], [7.0, 8.0]])
        )
        assert np.any(np.abs(noise_added) > 0)


def test_get_norm() -> None:
    """Test get_norm function."""
    # Prepare
    update = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]

    # Execute
    result = get_norm(update)

    expected = float(
        np.linalg.norm(np.concatenate([sub_update.flatten() for sub_update in update]))
    )

    # Assert
    assert expected == result


def test_clip_inputs_inplace() -> None:
    """Test clip_inputs_inplace function."""
    # Prepare
    updates = [
        np.array([[1.5, -0.5], [2.0, -1.0]]),
        np.array([0.5, -0.5]),
        np.array([[-0.5, 1.5], [-1.0, 2.0]]),
        np.array([-0.5, 0.5]),
    ]
    clipping_norm = 1.5

    original_updates = [np.copy(update) for update in updates]

    # Execute
    clip_inputs_inplace(updates, clipping_norm)

    # Assert
    for updated, original_update in zip(updates, original_updates, strict=True):
        clip_norm = np.linalg.norm(original_update)
        assert np.all(updated <= clip_norm) and np.all(updated >= -clip_norm)


def test_compute_stdv() -> None:
    """Test compute_stdv function."""
    # Prepare
    noise_multiplier = 1.0
    clipping_norm = 0.5
    num_sampled_clients = 10

    # Execute
    stdv = compute_stdv(noise_multiplier, clipping_norm, num_sampled_clients)

    # Assert
    expected_stdv = float((noise_multiplier * clipping_norm) / num_sampled_clients)
    assert stdv == expected_stdv


def test_compute_clip_model_update() -> None:
    """Test compute_clip_model_update function."""
    # Prepare
    param1 = [
        np.array([0.5, 1.5, 2.5]),
        np.array([3.5, 4.5, 5.5]),
        np.array([6.5, 7.5, 8.5]),
    ]
    param2 = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    ]
    clipping_norm = 4

    expected_result = [
        np.array([0.5, 1.5, 2.5]),
        np.array([3.5, 4.5, 5.5]),
        np.array([6.5, 7.5, 8.5]),
    ]

    # Execute
    compute_clip_model_update(param1, param2, clipping_norm)

    # Assert
    for i, param in enumerate(param1):
        np.testing.assert_array_almost_equal(param, expected_result[i])


def test_compute_adaptive_noise_params() -> None:
    """Test compute_adaptive_noise_params function."""
    # Test valid input with positive noise_multiplier
    noise_multiplier = 1.0
    num_sampled_clients = 100.0
    clipped_count_stddev = None
    result = compute_adaptive_noise_params(
        noise_multiplier, num_sampled_clients, clipped_count_stddev
    )

    # Assert
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] > 0.0
    assert result[1] > 0.0

    # Test valid input with zero noise_multiplier
    noise_multiplier = 0.0
    num_sampled_clients = 50.0
    clipped_count_stddev = None
    result = compute_adaptive_noise_params(
        noise_multiplier, num_sampled_clients, clipped_count_stddev
    )

    # Assert
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == 0.0
    assert result[1] == 0.0

    # Test valid input with specified clipped_count_stddev
    noise_multiplier = 3.0
    num_sampled_clients = 200.0
    clipped_count_stddev = 5.0
    result = compute_adaptive_noise_params(
        noise_multiplier, num_sampled_clients, clipped_count_stddev
    )

    # Assert
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == clipped_count_stddev
    assert result[1] > 0.0

    # Test invalid input with noise_multiplier >= 2 * clipped_count_stddev
    noise_multiplier = 10.0
    num_sampled_clients = 100.0
    clipped_count_stddev = None
    try:
        compute_adaptive_noise_params(
            noise_multiplier, num_sampled_clients, clipped_count_stddev
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError not raised.")

    # Test intermediate calculation
    noise_multiplier = 3.0
    num_sampled_clients = 200.0
    clipped_count_stddev = 5.0
    result = compute_adaptive_noise_params(
        noise_multiplier, num_sampled_clients, clipped_count_stddev
    )
    temp_value = (noise_multiplier ** (-2) - (2 * clipped_count_stddev) ** (-2)) ** -0.5

    # Assert
    assert np.isclose(result[1], temp_value, rtol=1e-6)


def test_differential_privacy_with_integer_arrays() -> None:
    """Test that integer arrays (e.g. BatchNorm num_batches_tracked) are not scaled."""
    # Prepare: float weights update and an integer buffer (like num_batches_tracked)
    float_array = np.array([3.0, 4.0], dtype=np.float32)  # norm is 5.0
    int_array = np.array([10], dtype=np.int64)
    updates = [float_array, int_array]

    # Test get_norm ignores integer arrays when float arrays exist
    norm = get_norm(updates)
    assert norm == 5.0

    # Test clip_inputs_inplace with clipping_norm = 2.5 (scaling factor 0.5)
    # Float updates should be scaled down by 0.5, integer updates should remain untouched
    clip_inputs_inplace(updates, clipping_norm=2.5)
    np.testing.assert_array_almost_equal(
        updates[0], np.array([1.5, 2.0], dtype=np.float32)
    )
    assert updates[1].dtype == np.int64
    assert updates[1][0] == 10

    # Test adaptive_clip_inputs_inplace
    float_array2 = np.array([3.0, 4.0], dtype=np.float32)
    int_array2 = np.array([20], dtype=np.int64)
    updates2 = [float_array2, int_array2]
    clipped = adaptive_clip_inputs_inplace(updates2, clipping_norm=2.5)
    assert clipped is True
    np.testing.assert_array_almost_equal(
        updates2[0], np.array([1.5, 2.0], dtype=np.float32)
    )
    assert updates2[1][0] == 20

    # Test add_gaussian_noise_inplace: noise is only added to floating point arrays
    add_gaussian_noise_inplace(updates2, std_dev=0.5)
    assert updates2[1][0] == 20


def test_zero_norm_counter_only_update() -> None:
    """Test zero norm does not cause ZeroDivisionError when only counter changes."""
    # Zero float updates (no parameter change) and changed integer counter
    float_array = np.array([0.0, 0.0], dtype=np.float32)
    int_array = np.array([5], dtype=np.int64)
    updates = [float_array, int_array]

    norm = get_norm(updates)
    assert norm == 0.0

    # clip_inputs_inplace should return gracefully without ZeroDivisionError
    clip_inputs_inplace(updates, clipping_norm=1.0)
    assert updates[0][0] == 0.0
    assert updates[1][0] == 5

    # adaptive_clip_inputs_inplace should return False
    clipped = adaptive_clip_inputs_inplace(updates, clipping_norm=1.0)
    assert clipped is False


