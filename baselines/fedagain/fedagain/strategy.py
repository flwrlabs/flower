"""fedagain: The FedAgain strategy (trust-weighted aggregation).

FedAgain (Reyes-Amezcua et al., 2026, Sec. 3.3) replaces the data-size weights of
FedAvg with trust weights built from two signals:

* the **benchmark error** ``E_k`` (Eq. 6): mean cross-entropy of the received
  global model on the client's local validation set, computed by the client
  *before* local training and sent to the server as a metric;
* the **parameter divergence** ``D_k`` (Eq. 7): L2 distance between the client's
  updated model and the global model it received, computed by the server.

The raw trust score is ``T_k = 1 / (E_k * D_k + eps)`` (Eq. 8), it is normalized
to sum to one (Eq. 9) and the new global model is the trust-weighted average of
the client models (Eq. 10). The data size of a client is not used.
"""

from collections.abc import Iterable, Sequence
from logging import INFO, WARNING

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from numpy.typing import NDArray

BENCHMARK_ERROR_KEY = "benchmark-error"
CLIENT_ID_KEY = "partition-id"


def floating_keys(arrays: dict[str, NDArray]) -> list[str]:
    """Return the keys of floating-point arrays.

    ``D_k`` is measured on floating-point tensors only. Integer buffers such as the
    ``num_batches_tracked`` counters of BatchNorm layers grow by the number of
    local mini-batches and would otherwise dominate the norm.
    """
    return [k for k, v in arrays.items() if np.issubdtype(v.dtype, np.floating)]


def parameter_divergence(
    client_arrays: dict[str, NDArray],
    global_arrays: dict[str, NDArray],
    keys: Sequence[str],
) -> float:
    """Compute ``D_k = || W_k - W_server ||_2`` over ``keys`` (Eq. 7), in float64."""
    sq_norm = 0.0
    for key in keys:
        diff = client_arrays[key].astype(np.float64) - global_arrays[key].astype(
            np.float64
        )
        sq_norm += float(np.sum(diff * diff))
    return float(np.sqrt(sq_norm))


def trust_weights(
    benchmark_errors: Sequence[float],
    divergences: Sequence[float],
    epsilon: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute normalized trust weights (Eqs. 8 and 9).

    Returns ``(weights, raw_scores)`` where ``raw_scores[k] = 1 / (E_k D_k + eps)``
    and ``weights`` sum to one. If any raw score is not finite or their sum is not
    positive, uniform weights are returned as a defensive fallback.
    """
    e_k = np.asarray(benchmark_errors, dtype=np.float64)
    d_k = np.asarray(divergences, dtype=np.float64)
    if e_k.shape != d_k.shape or e_k.ndim != 1 or e_k.size == 0:
        raise ValueError("benchmark_errors and divergences must be 1-D, same length")
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / (e_k * d_k + epsilon)
    total = raw.sum()
    if not np.all(np.isfinite(raw)) or not np.isfinite(total) or total <= 0:
        return np.full(e_k.size, 1.0 / e_k.size), raw
    return raw / total, raw


def weighted_average(
    arrays: Sequence[dict[str, NDArray]],
    weights: Sequence[float] | NDArray[np.float64],
    reference: dict[str, NDArray],
) -> dict[str, NDArray]:
    """Average client arrays with ``weights`` (Eq. 10).

    The sum is accumulated in float64 and cast back to the dtype of
    ``reference``; integer buffers are rounded.
    """
    out: dict[str, NDArray] = {}
    for key, ref in reference.items():
        acc = np.zeros(ref.shape, dtype=np.float64)
        for client, weight in zip(arrays, weights, strict=False):
            acc += float(weight) * client[key].astype(np.float64)
        if np.issubdtype(ref.dtype, np.floating):
            out[key] = acc.astype(ref.dtype)
        else:
            out[key] = np.rint(acc).astype(ref.dtype)
    return out


def _to_numpy_dict(record: ArrayRecord) -> dict[str, NDArray]:
    return {key: array.numpy() for key, array in record.items()}


class FedAgain(FedAvg):
    """FedAgain: trust-based aggregation with benchmark error and divergence.

    Clients must add their benchmark error ``E_k`` to the ``MetricRecord`` of
    their training reply under ``benchmark_error_key``. The strategy keeps a copy
    of the global model sent in each round to compute ``D_k`` on the server.

    Parameters
    ----------
    epsilon : float (default: 1e-3)
        Constant of Eq. 8 that prevents a division by zero.
    benchmark_error_key : str (default: "benchmark-error")
        Key of ``E_k`` in the clients' training ``MetricRecord``.
    client_id_key : str (default: "partition-id")
        Optional key used to label clients in the trust history.
    **kwargs
        Arguments forwarded to :class:`flwr.serverapp.strategy.FedAvg`.
        ``weighted_by_key`` is only used to aggregate metrics, never the model.
    """

    def __init__(
        self,
        *,
        epsilon: float = 1e-3,
        benchmark_error_key: str = BENCHMARK_ERROR_KEY,
        client_id_key: str = CLIENT_ID_KEY,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.benchmark_error_key = benchmark_error_key
        self.client_id_key = client_id_key
        self._global_arrays: dict[str, NDArray] | None = None
        self.trust_history: list[dict] = []

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedAgain:")
        log(INFO, "\t│\t├── epsilon: %s", self.epsilon)
        log(INFO, "\t│\t└── benchmark error key: '%s'", self.benchmark_error_key)
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Store the global model of this round and configure training."""
        # The copy is the reference for D_k; the flag asks clients for E_k
        self._global_arrays = _to_numpy_dict(arrays)
        config["compute-benchmark-error"] = True
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate client models with FedAgain trust weights."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None
        if self._global_arrays is None:
            raise RuntimeError("aggregate_train called before configure_train")

        # Collect E_k (client metric), the client model and its labels per reply
        contents = [msg.content for msg in valid_replies]
        client_arrays: list[dict[str, NDArray]] = []
        errors: list[float] = []
        client_ids: list[int] = []
        num_examples: list[float] = []
        for msg, content in zip(valid_replies, contents, strict=False):
            metrics = next(iter(content.metric_records.values()))
            if self.benchmark_error_key not in metrics:
                raise ValueError(
                    f"Reply from node {msg.metadata.src_node_id} has no "
                    f"'{self.benchmark_error_key}' metric. FedAgain clients must "
                    "report their benchmark error E_k."
                )
            errors.append(float(metrics[self.benchmark_error_key]))  # type: ignore
            client_ids.append(
                int(metrics.get(self.client_id_key, msg.metadata.src_node_id))  # type: ignore
            )
            num_examples.append(float(metrics.get(self.weighted_by_key, 0.0)))  # type: ignore
            client_arrays.append(
                _to_numpy_dict(next(iter(content.array_records.values())))
            )

        # D_k against the global model stored in configure_train (Eq. 7)
        keys = floating_keys(self._global_arrays)
        divergences = [
            parameter_divergence(arrs, self._global_arrays, keys)
            for arrs in client_arrays
        ]
        # Trust weights replace the data-size weights of FedAvg (Eqs. 8-10)
        weights, raw = trust_weights(errors, divergences, self.epsilon)
        if not np.all(np.isfinite(raw)) or raw.sum() <= 0:
            log(WARNING, "FedAgain: degenerate trust scores, using uniform weights")

        aggregated = weighted_average(client_arrays, weights, self._global_arrays)
        arrays = ArrayRecord({k: Array(np.asarray(v)) for k, v in aggregated.items()})

        # Keep every signal of the round so the ServerApp can save and plot it
        self.trust_history.append(
            {
                "round": server_round,
                "client_ids": client_ids,
                "num_examples": num_examples,
                "benchmark_error": errors,
                "divergence": divergences,
                "trust_score": raw.tolist(),
                "weight": weights.tolist(),
            }
        )
        order = np.argsort(client_ids)
        log(
            INFO,
            "FedAgain round %d weights (client: weight): %s",
            server_round,
            ", ".join(f"{client_ids[i]}: {weights[i]:.3f}" for i in order),
        )

        # Client metrics are still aggregated as in FedAvg (data-size weighted)
        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)
        metrics["trust-weight-min"] = float(weights.min())
        metrics["trust-weight-max"] = float(weights.max())
        return arrays, metrics
