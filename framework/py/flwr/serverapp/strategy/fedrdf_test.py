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
"""FedRDF tests."""


import numpy as np

from flwr.common import ArrayRecord

from .fedrdf import FedRDF
from .strategy_utils_test import create_mock_reply


def test_fedrdf_aggregate_train_with_threshold_zero() -> None:
    """Test aggregate_train with threshold=0 (always use FFT)."""
    # Prepare
    strategy = FedRDF(threshold=0.0)

    # Create mock replies with simple arrays
    weights0 = np.array([[1.0, 2.0]], dtype=np.float32)
    weights1 = np.array([[2.0, 3.0]], dtype=np.float32)
    weights2 = np.array([[3.0, 4.0]], dtype=np.float32)

    replies = [
        create_mock_reply(ArrayRecord([weights0]), num_examples=10),
        create_mock_reply(ArrayRecord([weights1]), num_examples=10),
        create_mock_reply(ArrayRecord([weights2]), num_examples=10),
    ]

    # Execute
    arrays_aggregated, metrics = strategy.aggregate_train(
        server_round=1, replies=replies
    )

    # Assert
    assert arrays_aggregated is not None
    assert metrics is not None

    # Verify aggregated arrays have correct shape
    aggregated_arrays = arrays_aggregated.to_numpy_ndarrays()
    assert len(aggregated_arrays) == 1
    assert aggregated_arrays[0].shape == (1, 2)


def test_fedrdf_aggregate_train_no_replies() -> None:
    """Test aggregate_train with no replies."""
    # Prepare
    strategy = FedRDF()

    # Execute
    arrays_aggregated, metrics = strategy.aggregate_train(
        server_round=1, replies=[]
    )

    # Assert
    assert arrays_aggregated is None
    assert metrics is None


def test_fedrdf_aggregate_train_with_high_threshold() -> None:
    """Test aggregate_train with high threshold (should use weighted FedAvg)."""
    # Prepare
    strategy = FedRDF(threshold=1.0)  # High threshold

    # Create mock replies with similar arrays (low skewness)
    weights0 = np.array([[1.0, 2.0]], dtype=np.float32)
    weights1 = np.array([[2.0, 3.0]], dtype=np.float32)
    weights2 = np.array([[3.0, 4.0]], dtype=np.float32)

    replies = [
        create_mock_reply(ArrayRecord([weights0]), num_examples=10),
        create_mock_reply(ArrayRecord([weights1]), num_examples=10),
        create_mock_reply(ArrayRecord([weights2]), num_examples=10),
    ]

    # Execute
    arrays_aggregated, _ = strategy.aggregate_train(
        server_round=1, replies=replies
    )

    # Assert
    assert arrays_aggregated is not None
    aggregated_arrays = arrays_aggregated.to_numpy_ndarrays()
    assert len(aggregated_arrays) == 1
    assert aggregated_arrays[0].shape == (1, 2)

    # With equal weights and low skewness, should be close to weighted average
    expected_avg = np.array([[2.0, 3.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(
        aggregated_arrays[0], expected_avg, decimal=1
    )


def test_fedrdf_ks_proportion() -> None:
    """Test _ks_proportion method."""
    # Prepare
    strategy = FedRDF()

    # Create a sample with normal distribution
    np.random.seed(42)
    sample = np.random.normal(0, 1, 100)

    # Execute
    proportion = strategy._ks_proportion(sample)

    # Assert - proportion should be between 0 and 1
    assert 0.0 <= proportion <= 1.0


def test_fedrdf_compute_skewness() -> None:
    """Test _compute_skewness method."""
    # Prepare
    strategy = FedRDF()

    # Create arrays with similar distributions (low skewness)
    np.random.seed(42)
    arrays = [np.random.normal(0, 1, (5, 5)) for _ in range(3)]

    # Execute
    skewness = strategy._compute_skewness(arrays)

    # Assert - skewness should be between 0 and 1
    assert 0.0 <= skewness <= 1.0


def test_fedrdf_fourier_aggregate() -> None:
    """Test _fourier_aggregate method."""
    # Prepare
    strategy = FedRDF()

    # Create simple arrays
    arrays = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    ]

    # Execute
    result = strategy._fourier_aggregate(arrays)

    # Assert
    assert result.shape == arrays[0].shape
    assert result.dtype == arrays[0].dtype


def test_fedrdf_with_poisoned_updates() -> None:
    """Test FedRDF robustness against poisoned updates."""
    # Prepare
    strategy = FedRDF(threshold=0.0)  # Always use FFT

    # Create benign and poisoned updates
    benign1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    benign2 = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=np.float32)
    benign3 = np.array([[0.9, 1.9], [2.9, 3.9]], dtype=np.float32)
    poisoned = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)  # Outlier

    replies = [
        create_mock_reply(ArrayRecord([benign1]), num_examples=10),
        create_mock_reply(ArrayRecord([benign2]), num_examples=10),
        create_mock_reply(ArrayRecord([benign3]), num_examples=10),
        create_mock_reply(ArrayRecord([poisoned]), num_examples=10),
    ]

    # Execute
    arrays_aggregated, _ = strategy.aggregate_train(server_round=1, replies=replies)

    # Assert - aggregated result should be closer to benign updates than poisoned
    assert arrays_aggregated is not None
    aggregated_arrays = arrays_aggregated.to_numpy_ndarrays()

    benign_mean = (benign1 + benign2 + benign3) / 3
    dist_to_benign = np.mean(np.abs(aggregated_arrays[0] - benign_mean))
    dist_to_poisoned = np.mean(np.abs(aggregated_arrays[0] - poisoned))

    # FedRDF should keep aggregation closer to benign clients
    assert dist_to_benign < dist_to_poisoned


def test_fedrdf_weighted_aggregation() -> None:
    """Test FedRDF with different weights (num_examples)."""
    # Prepare
    strategy = FedRDF(threshold=1.0)  # High threshold to use weighted FedAvg

    # Create arrays with different weights
    weights0 = np.array([[1.0, 2.0]], dtype=np.float32)
    weights1 = np.array([[2.0, 3.0]], dtype=np.float32)

    replies = [
        create_mock_reply(ArrayRecord([weights0]), num_examples=10),  # Weight: 10
        create_mock_reply(ArrayRecord([weights1]), num_examples=30),  # Weight: 30
    ]

    # Execute
    arrays_aggregated, _ = strategy.aggregate_train(server_round=1, replies=replies)

    # Assert - weighted average should favor the second client (3x weight)
    assert arrays_aggregated is not None
    aggregated_arrays = arrays_aggregated.to_numpy_ndarrays()

    # Expected weighted average: (1.0*10 + 2.0*30) / 40 = 1.75
    #                            (2.0*10 + 3.0*30) / 40 = 2.75
    expected = np.array([[1.75, 2.75]], dtype=np.float32)
    np.testing.assert_array_almost_equal(
        aggregated_arrays[0], expected, decimal=2
    )
