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
"""FedRDF: A Robust and Dynamic Aggregation Function Against Poisoning Attacks.

Reference:
E. Mármol Campos, A. González-Vidal, J. L. Hernández-Ramos, and A. Skarmeta,
"FedRDF: A Robust and Dynamic Aggregation Function Against Poisoning Attacks in
Federated Learning," IEEE Transactions on Emerging Topics in Computing, vol. 13,
no. 1, pp. 48–67, 2025. DOI: 10.1109/TETC.2024.3474484

Paper: https://ieeexplore.ieee.org/abstract/document/10713851
"""

from collections.abc import Callable, Iterable
from logging import INFO, WARNING

import numpy as np

# Import scipy with try/except for optional dependency
try:
    from scipy.stats import ks_2samp

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from flwr.common import (
    ArrayRecord,
    Message,
    MetricRecord,
    RecordDict,
    log,
)

from .fedavg import FedAvg


class FedRDF(FedAvg):
    """Robust and Dynamic Aggregation Function (FedRDF) strategy.

    FedRDF adaptively switches between standard FedAvg and FFT-based robust
    aggregation depending on the detected skewness in client updates. High
    skewness indicates potential poisoning attacks, triggering the robust
    FFT aggregation mechanism.

    The strategy analyzes client weight distributions using statistical tests
    (Kolmogorov-Smirnov) to compute skewness. When skewness exceeds a threshold,
    it applies Fourier Transform-based aggregation to mitigate the impact of
    poisoned updates.

    Parameters
    ----------
    fraction_train : float (default: 1.0)
        Fraction of nodes used during training. In case `min_train_nodes`
        is larger than `fraction_train * total_connected_nodes`, `min_train_nodes`
        will still be sampled.
    fraction_evaluate : float (default: 1.0)
        Fraction of nodes used during validation. In case `min_evaluate_nodes`
        is larger than `fraction_evaluate * total_connected_nodes`,
        `min_evaluate_nodes` will still be sampled.
    min_train_nodes : int (default: 2)
        Minimum number of nodes used during training.
    min_evaluate_nodes : int (default: 2)
        Minimum number of nodes used during validation.
    min_available_nodes : int (default: 2)
        Minimum number of total nodes in the system.
    weighted_by_key : str (default: "num-examples")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages for both ArrayRecords and MetricRecords.
    arrayrecord_key : str (default: "arrays")
        Key used to store the ArrayRecord when constructing Messages.
    configrecord_key : str (default: "config")
        Key used to store the ConfigRecord when constructing Messages.
    train_metrics_aggr_fn : callable | None (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    evaluate_metrics_aggr_fn : callable | None (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from evaluation round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    threshold : float (default: 0.0)
        Skewness threshold for switching between FedAvg and FFT aggregation.
        - If threshold <= 0: Always use FFT-based robust aggregation
        - If threshold > 0: Use FFT only when detected skewness > threshold

    Reference
    ---------
    E. Mármol Campos et al., "FedRDF: A Robust and Dynamic Aggregation Function
    Against Poisoning Attacks in Federated Learning," IEEE TETC, 2025.

    Raises
    ------
    ImportError
        If scipy is not installed. Install with: pip install scipy>=1.7.0
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-positional-arguments
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        threshold: float = 0.0,
    ) -> None:
        """Initialize FedRDF strategy."""
        # Check scipy dependency
        if not HAS_SCIPY:
            raise ImportError(
                "FedRDF requires scipy to be installed. "
                "Install it with: pip install scipy>=1.7.0"
            )

        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )
        self.threshold = threshold

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords using FedRDF.

        This method applies the FedRDF robust aggregation algorithm to detect
        and mitigate poisoning attacks by analyzing the skewness of client updates.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        replies : Iterable[Message]
            Messages containing ArrayRecords and MetricRecords from clients.

        Returns
        -------
        arrays : ArrayRecord | None
            Aggregated ArrayRecord with robust model parameters, or None if aggregation failed.
        metrics : MetricRecord | None
            Aggregated MetricRecord with training metrics.
        """
        # Check and log replies
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

        # Extract content from valid messages
        reply_contents = [msg.content for msg in valid_replies]

        # Extract arrays and weights from RecordDicts
        arrays_and_weights = []
        for record_dict in reply_contents:
            arrayrecord = record_dict.get(self.arrayrecord_key)
            if arrayrecord is None:
                log(WARNING, "ArrayRecord not found in message, skipping")
                continue

            # Get weight (num_examples) from metrics
            weight = 1.0  # Default weight
            if "metrics" in record_dict:
                metrics = record_dict["metrics"]
                if isinstance(metrics, MetricRecord):
                    weight = float(metrics.get(self.weighted_by_key, 1.0))

            arrays_and_weights.append((arrayrecord, weight))

        if not arrays_and_weights:
            log(WARNING, "No valid arrays found in replies")
            return None, None

        # Perform FedRDF aggregation
        aggregated_arrayrecord = self._aggregate_fedrdf(
            arrays_and_weights, server_round
        )

        # Aggregate metrics using parent's method
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        return aggregated_arrayrecord, metrics

    def _aggregate_fedrdf(
        self,
        arrays_and_weights: list[tuple[ArrayRecord, float]],
        server_round: int,
    ) -> ArrayRecord:
        """Perform FedRDF aggregation with adaptive switching.

        Computes skewness in client updates and adaptively chooses between:
        - Standard weighted averaging (FedAvg) for normal scenarios
        - FFT-based robust aggregation for high-skewness scenarios (attacks)

        Parameters
        ----------
        arrays_and_weights : list[tuple[ArrayRecord, float]]
            List of (ArrayRecord, weight) tuples from clients.
        server_round : int
            Current server round (for logging).

        Returns
        -------
        ArrayRecord
            Aggregated ArrayRecord with robust model weights.
        """
        # Extract all keys from the first ArrayRecord
        first_record = arrays_and_weights[0][0]
        keys = list(first_record.keys())

        # Convert ArrayRecords to dictionaries of numpy arrays
        arrays_dict_list = []
        weights_list = []
        for arrayrecord, weight in arrays_and_weights:
            # Convert ArrayRecord to dict of numpy arrays, preserving keys
            arrays_dict = {k: arrayrecord[k].numpy() for k in keys}
            arrays_dict_list.append(arrays_dict)
            weights_list.append(weight)

        # Always use FFT if threshold <= 0
        if self.threshold <= 0:
            log(
                INFO,
                "FedRDF round %d: Using FFT-based robust aggregation (threshold=%.4f)",
                server_round,
                self.threshold,
            )
            aggregated_dict = {}
            for key in keys:
                layer_arrays = [arr_dict[key] for arr_dict in arrays_dict_list]
                aggregated_dict[key] = self._fourier_aggregate(layer_arrays)

            from flwr.common import Array
            return ArrayRecord({k: Array(np.asarray(v)) for k, v in aggregated_dict.items()})

        # Compute skewness for each layer
        skewness_scores = []
        for key in keys:
            layer_arrays = [arr_dict[key] for arr_dict in arrays_dict_list]
            skewness = self._compute_skewness(layer_arrays)
            skewness_scores.append(skewness)

        avg_skewness = np.mean(skewness_scores)

        log(
            INFO,
            "FedRDF round %d: Detected skewness=%.4f, threshold=%.4f",
            server_round,
            avg_skewness,
            self.threshold,
        )

        # Adaptive decision based on skewness
        if avg_skewness < self.threshold:
            # Low skewness: Use standard weighted FedAvg
            log(
                INFO,
                "FedRDF round %d: Using weighted FedAvg (low skewness)",
                server_round,
            )
            total_weight = sum(weights_list)
            aggregated_dict = {}

            for key in keys:
                weighted_sum = sum(
                    arr_dict[key] * weight
                    for arr_dict, weight in zip(arrays_dict_list, weights_list, strict=True)
                )
                aggregated_dict[key] = weighted_sum / total_weight

        else:
            # High skewness: Use FFT-based robust aggregation
            log(
                INFO,
                "FedRDF round %d: Using FFT aggregation (high skewness detected)",
                server_round,
            )
            aggregated_dict = {}
            for key in keys:
                layer_arrays = [arr_dict[key] for arr_dict in arrays_dict_list]
                aggregated_dict[key] = self._fourier_aggregate(layer_arrays)

        from flwr.common import Array
        return ArrayRecord({k: Array(np.asarray(v)) for k, v in aggregated_dict.items()})

    def _ks_proportion(self, sample: np.ndarray) -> float:
        """Estimate distribution divergence using Kolmogorov-Smirnov test.

        Repeatedly samples random subsets from the data and tests if they come
        from the same distribution as the remaining data. Returns the proportion
        of tests that reject the null hypothesis (indicating divergence).

        Parameters
        ----------
        sample : np.ndarray
            Array of values to test for distribution divergence.

        Returns
        -------
        float
            Proportion of K-S tests that detected significant divergence (0-1).
        """
        # K-S test requires sufficient sample size (minimum ~5 samples per group)
        # If sample is too small, return 0.0 (no divergence detected)
        if len(sample) < 10:
            return 0.0

        total = []
        significance_level = 0.05

        for _ in range(100):  # Reduced from 1000 to 100 for performance
            # Randomly select 30% of the sample
            n_random = int(len(sample) * 0.3)
            # Ensure both groups have at least 3 samples for valid K-S test
            if n_random < 3 or len(sample) - n_random < 3:
                continue

            random_indices = np.random.choice(
                len(sample), size=n_random, replace=False
            )
            random_points = sample[random_indices]
            generated_points = np.delete(sample, random_indices)

            # Perform Kolmogorov-Smirnov test
            _, p_value = ks_2samp(generated_points, random_points)

            # Record if distributions are significantly different
            total.append(1 if p_value < significance_level else 0)

        # If no valid tests were performed, return 0.0
        return float(np.mean(total)) if total else 0.0

    def _compute_skewness(self, arrays: list[np.ndarray]) -> float:
        """Measure the skewness proportion across client weight arrays.

        Analyzes the distribution of weight values at each position across clients.
        High skewness indicates that some weights have significantly different
        distributions, which may signal poisoning attacks.

        Parameters
        ----------
        arrays : list[np.ndarray]
            List of weight arrays from different clients (same shape).

        Returns
        -------
        float
            Average skewness proportion across all weight positions (0-1).
        """
        # Stack arrays into matrix: (n_clients, n_weights)
        stacked = np.stack([arr.flatten() for arr in arrays], axis=0)

        # Standardize each weight position across clients
        means = stacked.mean(axis=0)
        stds = stacked.std(axis=0)
        # Avoid division by zero for constant weights
        stds = np.where(stds == 0, 1.0, stds)
        standardized = (stacked - means) / stds

        # Sample weight positions to avoid computational explosion with large models
        # Testing all positions is impractical for models with millions of parameters
        n_positions = standardized.shape[1]
        max_samples = 100  # Maximum positions to test (reduced for performance)
        if n_positions > max_samples:
            # Randomly sample positions
            sample_indices = np.random.choice(n_positions, size=max_samples, replace=False)
        else:
            sample_indices = np.arange(n_positions)

        # Compute K-S proportion for sampled weight positions
        proportions = []
        for i in sample_indices:
            prop = self._ks_proportion(standardized[:, i])
            proportions.append(prop)

        return float(np.mean(proportions))

    def _fourier_aggregate(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Aggregate weights using FFT-based robust aggregation.

        For each weight position, collects values across all clients, computes
        the FFT, and selects the value corresponding to the dominant frequency
        component. This approach is robust to outliers in the frequency domain.

        Parameters
        ----------
        arrays : list[np.ndarray]
            List of weight arrays from different clients (same shape).

        Returns
        -------
        np.ndarray
            Aggregated weight array with the same shape as input arrays.
        """
        result = np.zeros_like(arrays[0])
        flat_size = arrays[0].size

        for i in range(flat_size):
            # Collect values at position i from all clients
            values = np.array([arr.flat[i] for arr in arrays])

            # Compute FFT to analyze frequency components
            fft_values = np.fft.fft(values)

            # Find the dominant frequency component
            magnitudes = np.abs(fft_values)
            dominant_idx = int(np.argmax(magnitudes))

            # Use the original value corresponding to the dominant frequency
            result.flat[i] = values[dominant_idx]

        return result
