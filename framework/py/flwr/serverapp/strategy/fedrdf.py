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
from logging import INFO

import numpy as np

# Import scipy with try/except for optional dependency
try:
    from scipy.stats import ks_2samp

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.common import NDArray, log

from .fedavg import FedAvg
from .strategy_utils import aggregate_arrayrecords

# Number of harmonics and Gaussian smoothing bandwidth (as a fraction of the
# unit period) used by the Fourier-based density estimation, and the number of
# coordinates processed per chunk to bound peak memory
_FOURIER_MODES = 8
_FOURIER_BANDWIDTH = 0.05
_FOURIER_CHUNK = 65536


class FedRDF(FedAvg):
    """Robust and Dynamic Aggregation Function (FedRDF) strategy.

    FedRDF adaptively switches between standard FedAvg and FFT-based robust
    aggregation depending on the detected skewness in client updates. High
    skewness indicates potential poisoning attacks, triggering the robust
    FFT aggregation mechanism.

    The strategy analyzes client weight distributions using statistical tests
    (Kolmogorov-Smirnov) to compute skewness. When skewness reaches a threshold,
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
        - If threshold > 0: Use FFT only when the detected skewness reaches the
          threshold. The Kolmogorov-Smirnov based detection requires at least
          10 replies per round; with fewer replies the detected skewness is 0.0
          and FedAvg is used.

    Reference
    ---------
    E. Mármol Campos et al., "FedRDF: A Robust and Dynamic Aggregation Function
    Against Poisoning Attacks in Federated Learning," IEEE TETC, 2025.

    Raises
    ------
    ImportError
        If ``threshold > 0`` (adaptive mode, which uses the Kolmogorov-Smirnov
        test) and scipy is not installed. Install with: pip install scipy>=1.7.0
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
        # scipy is only used by the K-S test in adaptive mode (threshold > 0)
        if threshold > 0 and not HAS_SCIPY:
            raise ImportError(
                "FedRDF with threshold > 0 (adaptive mode) requires scipy for the "
                "Kolmogorov-Smirnov test. Install it with: pip install scipy>=1.7.0"
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
        self._rng = np.random.default_rng()

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
            Aggregated ArrayRecord with robust model parameters, or None if
            aggregation failed.
        metrics : MetricRecord | None
            Aggregated MetricRecord with training metrics.
        """
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

        # Replies are already validated to hold one ArrayRecord and one MetricRecord
        reply_contents = [msg.content for msg in valid_replies]

        aggregated_arrayrecord = self._aggregate_fedrdf(reply_contents, server_round)

        # Aggregate metrics using parent's method
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        return aggregated_arrayrecord, metrics

    def _aggregate_fedrdf(
        self,
        records: list[RecordDict],
        server_round: int,
    ) -> ArrayRecord:
        """Aggregate layers via Fourier-based selection or weighted FedAvg."""
        arrayrecords = [next(iter(record.array_records.values())) for record in records]
        keys = list(arrayrecords[0].keys())
        layers: dict[str, list[NDArray]] = {
            key: [arrayrecord[key].numpy() for arrayrecord in arrayrecords]
            for key in keys
        }

        use_fourier = self.threshold <= 0
        if not use_fourier:
            skewness = float(np.mean([self._compute_skewness(layers[k]) for k in keys]))
            use_fourier = skewness >= self.threshold
            log(
                INFO,
                "FedRDF round %d: skewness=%.4f, threshold=%.4f",
                server_round,
                skewness,
                self.threshold,
            )

        if not use_fourier:
            return aggregate_arrayrecords(records, self.weighted_by_key)

        return ArrayRecord({k: Array(self._fourier_aggregate(layers[k])) for k in keys})

    def _ks_proportion(self, sample: NDArray) -> float:
        """Return the proportion of K-S tests that detect divergence in a sample."""
        if not HAS_SCIPY:
            raise ImportError(
                "FedRDF with threshold > 0 (adaptive mode) requires scipy for the "
                "Kolmogorov-Smirnov test. Install it with: pip install scipy>=1.7.0"
            )
        if len(sample) < 10:
            return 0.0

        total = []
        for _ in range(100):
            n_random = int(len(sample) * 0.3)
            if n_random < 3 or len(sample) - n_random < 3:
                continue
            random_indices = self._rng.choice(len(sample), size=n_random, replace=False)
            random_points = sample[random_indices]
            generated_points = np.delete(sample, random_indices)
            _, p_value = ks_2samp(generated_points, random_points)
            total.append(1 if p_value < 0.05 else 0)

        return float(np.mean(total)) if total else 0.0

    def _compute_skewness(self, arrays: list[NDArray]) -> float:
        """Measure the skewness proportion across client weight arrays."""
        # Sample positions first to avoid materializing every parameter of
        # large models when only a subset is tested
        n_positions = arrays[0].size
        if n_positions > 100:
            sample_indices = self._rng.choice(n_positions, size=100, replace=False)
        else:
            sample_indices = np.arange(n_positions)

        stacked = np.stack([np.ravel(arr)[sample_indices] for arr in arrays], axis=0)
        means = stacked.mean(axis=0)
        stds = np.where(stacked.std(axis=0) == 0, 1.0, stacked.std(axis=0))
        standardized = (stacked - means) / stds

        proportions = [
            self._ks_proportion(standardized[:, i])
            for i in range(standardized.shape[1])
        ]
        return float(np.mean(proportions))

    def _fourier_aggregate(self, arrays: list[NDArray]) -> NDArray:
        """Select per coordinate the client weight exhibiting the highest density.

        Following Eq. 11 of the paper, the sorted per-coordinate client weights
        are projected into the frequency domain to ascertain their density
        function (Nanbu, 1995), and the weight where the estimated density
        peaks is selected, so weights isolated from the high-density regions
        (malicious clients) are excluded.
        """
        stacked = np.sort(np.stack([arr.ravel() for arr in arrays]), axis=0)
        low, span = stacked[0], stacked[-1] - stacked[0]
        span = np.where(span == 0, 1.0, span)

        damping = np.exp(
            -2.0 * (np.pi * np.arange(1, _FOURIER_MODES + 1) * _FOURIER_BANDWIDTH) ** 2
        )

        result = np.empty(stacked.shape[1], dtype=stacked.dtype)
        for start in range(0, stacked.shape[1], _FOURIER_CHUNK):
            chunk = slice(start, start + _FOURIER_CHUNK)
            values = stacked[:, chunk]
            # Map into [1/4, 3/4] of the unit period so the smallest and
            # largest weights do not become periodic neighbours
            phase = 0.25 + 0.5 * (values - low[chunk]) / span[chunk]
            base = np.exp(-2j * np.pi * phase)
            harmonics = base.copy()
            density = np.zeros(phase.shape)
            for weight in damping:
                density += weight * (harmonics.mean(axis=0) * harmonics.conj()).real
                harmonics *= base
            result[chunk] = values[
                np.argmax(density, axis=0), np.arange(values.shape[1])
            ]

        return result.reshape(arrays[0].shape)
