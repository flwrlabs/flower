"""floco: A Flower Baseline."""

from collections.abc import Iterable
from logging import INFO

import numpy as np
from flwr.app import MessageType
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
    log,
    ndarray_to_bytes,
)
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    sample_nodes,
)
from sklearn import decomposition


class Floco(FedAvg):
    r"""Federated Optimization strategy.

    Implementation based on https://openreview.net/pdf?id=JL2eMCfDW8

    Parameters
    ----------
    fraction_train : float, optional
        Fraction of nodes used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of nodes used during validation. Defaults to 1.0.
    min_train_nodes : int, optional
        Minimum number of nodes used during training. Defaults to 2.
    min_evaluate_nodes : int, optional
        Minimum number of nodes used during validation. Defaults to 2.
    min_available_nodes : int, optional
        Minimum number of total nodes in the system. Defaults to 2.
    tau : int
        Round at which to start projection.
    rho : float
        Radius of the ball around each projected client parameters
        from which models are sampled.
    endpoints : int
        Number of endpoints of the solution simplex.
    """

    def __init__(
        self,
        *,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        tau: int = 0,
        rho: float = 1.0,
        endpoints: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            **kwargs,
        )
        self.tau = tau
        self.rho = rho
        self.endpoints = endpoints
        # node_id -> pseudo-gradient
        self.client_gradients: dict[int, np.ndarray] = {}
        # node_id -> simplex projection coordinates
        self.client_subregion_parameters: dict[int, np.ndarray] = {}
        # Track which node_ids were sampled with the regular fraction
        self.last_selected_node_ids: list[int] = []
        # Store arrays from previous round for pseudo-gradient computation
        self.initial_arrays: ArrayRecord | None = None

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        if self.fraction_train == 0.0:
            return []

        # Compute regular sample size
        all_node_ids = list(grid.get_node_ids())
        num_nodes = int(len(all_node_ids) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)

        # Sample nodes for the regular round
        node_ids, _ = sample_nodes(grid, self.min_available_nodes, sample_size)
        self.last_selected_node_ids = list(node_ids)

        if (server_round + 1) == self.tau:
            # Round before projection: sample ALL nodes to get gradients
            all_sample_size = max(len(all_node_ids), self.min_available_nodes)
            node_ids, _ = sample_nodes(grid, self.min_available_nodes, all_sample_size)

        elif server_round == self.tau:
            # Round of projection: compute projections from collected gradients
            projected = project_clients(self.client_gradients, self.endpoints)
            self.client_subregion_parameters = dict(
                zip(sorted(self.client_gradients.keys()), projected)
            )

        config["server-round"] = server_round

        if server_round >= self.tau:
            # Per-node messages with unique subregion parameters
            messages = []
            for node_id in node_ids:
                node_config = ConfigRecord(dict(config))
                if node_id in self.client_subregion_parameters:
                    node_config["center"] = ndarray_to_bytes(
                        self.client_subregion_parameters[node_id]
                    )
                    node_config["radius"] = self.rho
                record = RecordDict(
                    {self.arrayrecord_key: arrays, self.configrecord_key: node_config}
                )
                messages.append(
                    Message(
                        content=record,
                        message_type=MessageType.TRAIN,
                        dst_node_id=node_id,
                    )
                )
            log(INFO, "configure_train: Sent %s messages (round %s)", len(messages), server_round)
            return messages

        # Before projection: uniform messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        log(INFO, "configure_train: Sampled %s nodes (round %s)", len(node_ids), server_round)
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate training results."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

        if (server_round + 1) == self.tau:
            # Compute pseudo-gradients from all clients
            filtered_replies = []
            for msg in valid_replies:
                node_id = msg.metadata.src_node_id
                received_arrays = next(iter(msg.content.array_records.values()))
                # Compute pseudo-gradient: diff between initial and received
                if self.initial_arrays is not None:
                    client_grads = []
                    keys = list(received_arrays.keys())
                    for key in keys[-self.endpoints:]:
                        init_arr = self.initial_arrays[key].numpy().flatten()
                        recv_arr = received_arrays[key].numpy().flatten()
                        client_grads.append(init_arr - recv_arr)
                    self.client_gradients[node_id] = np.concatenate(client_grads)

                # Only keep regularly-sampled nodes for aggregation
                if node_id in self.last_selected_node_ids:
                    filtered_replies.append(msg)

            self.client_gradients = {
                k: self.client_gradients[k]
                for k in sorted(self.client_gradients.keys())
            }
            valid_replies = filtered_replies

        if not valid_replies:
            return None, None

        reply_contents = [msg.content for msg in valid_replies]

        # Aggregate ArrayRecords
        arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        # Store for next round's gradient computation
        self.initial_arrays = arrays

        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, _ = sample_nodes(grid, self.min_available_nodes, sample_size)

        config["server-round"] = server_round

        if server_round >= self.tau:
            # Per-node messages with unique subregion parameters
            messages = []
            for node_id in node_ids:
                node_config = ConfigRecord(dict(config))
                if node_id in self.client_subregion_parameters:
                    node_config["center"] = ndarray_to_bytes(
                        self.client_subregion_parameters[node_id]
                    )
                    node_config["radius"] = self.rho
                record = RecordDict(
                    {self.arrayrecord_key: arrays, self.configrecord_key: node_config}
                )
                messages.append(
                    Message(
                        content=record,
                        message_type=MessageType.EVALUATE,
                        dst_node_id=node_id,
                    )
                )
            log(INFO, "configure_evaluate: Sent %s messages (round %s)", len(messages), server_round)
            return messages

        # Before projection: uniform messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        log(INFO, "configure_evaluate: Sampled %s nodes (round %s)", len(node_ids), server_round)
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)


def project_clients(client_gradients, endpoints):
    """Optimize client projection onto a simplex of dimension endpoints-1."""
    client_stats = np.array(list(client_gradients.values()))
    kappas = decomposition.PCA(n_components=endpoints).fit_transform(client_stats)
    # Find optimal projection
    lowest_log_energy = np.inf
    best_beta = None
    for z in np.linspace(1e-4, 1, 1000):
        betas = _project_client_onto_simplex(kappas, z=z)
        betas /= betas.sum(axis=1, keepdims=True)
        log_energy = _riesz_s_energy(betas)
        if log_energy not in [-np.inf, np.inf] and log_energy < lowest_log_energy:
            lowest_log_energy = log_energy
            best_beta = betas
    return best_beta


def _project_client_onto_simplex(kappas, z):
    """Project clients onto a simplex of dimension endpoints-1."""
    sorted_kappas = np.sort(kappas, axis=1)[:, ::-1]
    z = np.ones(len(kappas)) * z
    cssv = np.cumsum(sorted_kappas, axis=1) - z[:, np.newaxis]
    ind = np.arange(kappas.shape[1]) + 1
    cond = sorted_kappas - cssv / ind > 0
    nonzero = np.count_nonzero(cond, axis=1)
    normalized_kappas = cssv[np.arange(len(kappas)), nonzero - 1] / nonzero
    betas = np.maximum(kappas - normalized_kappas[:, np.newaxis], 0)
    return betas


def _riesz_s_energy(simplex_points):
    """Compute Riesz s-energy of client projections.

    (https://www.sciencedirect.com/science/article/pii/S0021904503000315)
    """
    diff = simplex_points[:, None] - simplex_points[None, :]
    distance = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    epsilon = 1e-4  # epsilon is the smallest distance possible to avoid overflow
    distance[distance < epsilon] = epsilon
    # select only upper triangular matrix to have each mutual distance once
    mutual_dist = distance[np.triu_indices(len(simplex_points), 1)]
    mutual_dist[np.argwhere(mutual_dist == 0).flatten()] = epsilon
    energies = 1 / mutual_dist**2
    energy = energies[~np.isnan(energies)].sum()
    log_energy = -np.log(len(mutual_dist)) + np.log(energy)
    return log_energy
