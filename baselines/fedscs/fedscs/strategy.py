"""FedSCS strategy for Flower."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from flwr.common import Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Strategy


class FedSCS(Strategy):
    """Federated Stable Cosine Similarity aggregation strategy."""

    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 10,
        min_evaluate_nodes: int = 10,
        min_available_nodes: int = 10,
        epsilon: float = 1e-6,
        beta: float = 0.9,
    ) -> None:
        """Initialize the FedSCS strategy."""
        if not 0.0 < fraction_train <= 1.0:
            raise ValueError("fraction_train must be in (0, 1].")

        if not 0.0 < fraction_evaluate <= 1.0:
            raise ValueError("fraction_evaluate must be in (0, 1].")

        if min_train_nodes < 1:
            raise ValueError("min_train_nodes must be at least 1.")

        if min_evaluate_nodes < 1:
            raise ValueError("min_evaluate_nodes must be at least 1.")

        if min_available_nodes < 1:
            raise ValueError("min_available_nodes must be at least 1.")

        if epsilon <= 0.0:
            raise ValueError("epsilon must be greater than 0.")

        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1).")

        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_nodes = min_train_nodes
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.epsilon = epsilon
        self.beta = beta

        # FedAvg is used only for Flower's node sampling and message
        # configuration. FedSCS performs the actual model aggregation.
        self._sampling_strategy = FedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
        )

        # Per-client temporal history.
        self.client_histories: dict[int, dict[str, list[float]]] = {}

        # Global model immediately before the current local-training round.
        self._global_state: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _flatten_update(
        update: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Flatten a model update into a single vector."""
        return torch.cat(
            [
                value.detach().float().flatten()
                for value in update.values()
            ]
        )

    @staticmethod
    def _cosine_similarity(
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> float:
        """Compute a non-negative cosine similarity."""
        u = u.detach().float()
        v = v.detach().float()

        if not torch.isfinite(u).all() or not torch.isfinite(v).all():
            return 0.0

        u_norm = torch.linalg.vector_norm(u)
        v_norm = torch.linalg.vector_norm(v)

        if u_norm.item() <= 0.0 or v_norm.item() <= 0.0:
            return 0.0

        similarity = F.cosine_similarity(
            u.unsqueeze(0),
            v.unsqueeze(0),
            dim=1,
        ).item()

        if not torch.isfinite(torch.tensor(similarity)):
            return 0.0

        return max(float(similarity), 0.0)

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure clients for local training."""
        self._global_state = {
            key: value.detach().cpu().clone()
            for key, value in arrays.to_torch_state_dict().items()
        }

        return self._sampling_strategy.configure_train(
            server_round,
            arrays,
            config,
            grid,
        )

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure clients for evaluation."""
        return self._sampling_strategy.configure_evaluate(
            server_round,
            arrays,
            config,
            grid,
        )

    def _fedscs_aggregation(
        self,
        initial_state: dict[str, torch.Tensor],
        client_states: list[dict[str, torch.Tensor]],
        client_ids: list[int],
        server_round: int,
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Aggregate client models using FedSCS trust weights."""
        num_clients = len(client_states)

        if num_clients == 0:
            raise ValueError("FedSCS received no valid client updates.")

        if len(client_ids) != num_clients:
            raise ValueError("Client IDs and client states have different sizes.")

        # ---------------------------------------------------------------
        # Step 1: Compute local model updates.
        #
        # Delta_i = W_i - W_t
        # ---------------------------------------------------------------
        delta_states: list[dict[str, torch.Tensor]] = []

        for client_state in client_states:
            if set(client_state) != set(initial_state):
                raise ValueError(
                    "Client model parameters do not match the global model."
                )

            delta_states.append(
                {
                    key: (
                        client_state[key].detach().cpu().float()
                        - initial_state[key].detach().cpu().float()
                    )
                    for key in initial_state
                }
            )

        # ---------------------------------------------------------------
        # Step 2: Flatten updates.
        # ---------------------------------------------------------------
        flat_updates = [
            self._flatten_update(delta)
            for delta in delta_states
        ]

        if num_clients == 1:
            return client_states[0], [1.0]

        # ---------------------------------------------------------------
        # Step 3: Construct leave-one-out peer references.
        #
        # P_i = sum_{j != i} Delta_j
        # ---------------------------------------------------------------
        total_update = torch.stack(flat_updates).sum(dim=0)

        peer_sums = [
            total_update - flat_updates[i]
            for i in range(num_clients)
        ]

        # ---------------------------------------------------------------
        # Step 4: Compute stable cosine similarity.
        #
        # rho_i = max(cos(Delta_i, P_i), 0)
        # ---------------------------------------------------------------
        rho = [
            self._cosine_similarity(
                flat_updates[i],
                peer_sums[i],
            )
            for i in range(num_clients)
        ]

        # ---------------------------------------------------------------
        # Step 5: Temporal smoothing.
        #
        # s_i(t) = beta * s_i(t-1)
        #          + (1-beta) * rho_i(t)
        #
        # A client's history is updated only when that client
        # participates in a training round.
        # ---------------------------------------------------------------
        trust_scores: list[float] = []

        for client_id, rho_i in zip(client_ids, rho):
            history = self.client_histories.setdefault(
                client_id,
                {
                    "rho": [],
                    "s": [],
                    "nu": [],
                    "s_tilde": [],
                    "a": [],
                    "rounds": [],
                },
            )

            if history["s"]:
                previous_s = history["s"][-1]
                s_i = (
                    self.beta * previous_s
                    + (1.0 - self.beta) * rho_i
                )
            else:
                s_i = rho_i

            history["rho"].append(rho_i)
            history["s"].append(s_i)
            history["rounds"].append(float(server_round))

            trust_scores.append(s_i)

        # ---------------------------------------------------------------
        # Step 6: Temporal variation correction.
        #
        # nu_i(t) =
        #     |s_i(t) - s_i(t-1)| / (s_i(t-1) + epsilon)
        #
        # s_tilde_i(t) = s_i(t) / (1 + nu_i(t))
        # ---------------------------------------------------------------
        corrected_scores: list[float] = []

        for client_id, s_i in zip(client_ids, trust_scores):
            history = self.client_histories[client_id]

            if len(history["s"]) > 1:
                previous_s = history["s"][-2]
            else:
                previous_s = s_i

            variation = abs(s_i - previous_s) / (
                previous_s + self.epsilon
            )

            corrected_score = s_i / (1.0 + variation)

            history["nu"].append(variation)
            history["s_tilde"].append(corrected_score)

            corrected_scores.append(corrected_score)

        # ---------------------------------------------------------------
        # Step 7: Normalize trust scores into convex aggregation weights.
        # ---------------------------------------------------------------
        total_score = sum(corrected_scores)

        if (
            total_score <= self.epsilon
            or not torch.isfinite(torch.tensor(total_score))
        ):
            aggregation_weights = [
                1.0 / num_clients
            ] * num_clients
        else:
            aggregation_weights = [
                score / total_score
                for score in corrected_scores
            ]

        for client_id, weight in zip(
            client_ids,
            aggregation_weights,
        ):
            self.client_histories[client_id]["a"].append(weight)

        # ---------------------------------------------------------------
        # Step 8: Weighted model aggregation.
        #
        # W_(t+1) = sum_i a_i W_i
        # ---------------------------------------------------------------
        aggregated_state: dict[str, torch.Tensor] = {}

        for key, initial_value in initial_state.items():
            aggregated_value = sum(
                aggregation_weights[i]
                * client_states[i][key].detach().cpu().float()
                for i in range(num_clients)
            )

            if torch.is_floating_point(initial_value):
                aggregated_state[key] = aggregated_value.to(
                    dtype=initial_value.dtype
                )
            else:
                aggregated_state[key] = aggregated_value.to(
                    dtype=initial_value.dtype
                )

        # ---------------------------------------------------------------
        # Logging.
        # ---------------------------------------------------------------
        print(f"\n[FedSCS] Round {server_round}")
        print(f"  Participating clients: {num_clients}")
        print(f"  Beta: {self.beta:.4f}")
        print(f"  Epsilon: {self.epsilon:.2e}")

        for client_id, rho_i, score, weight in zip(
            client_ids,
            rho,
            corrected_scores,
            aggregation_weights,
        ):
            print(
                f"  Client {client_id}: "
                f"rho={rho_i:.4f}, "
                f"corrected_score={score:.4f}, "
                f"trust_weight={weight:.4f}"
            )

        return aggregated_state, aggregation_weights

    @staticmethod
    def _weighted_average(
        values: list[float],
        weights: list[int],
    ) -> float:
        """Compute a sample-weighted average."""
        if not values or not weights:
            return 0.0

        total_weight = sum(weights)

        if total_weight <= 0:
            return sum(values) / len(values)

        return sum(
            value * weight
            for value, weight in zip(values, weights)
        ) / total_weight

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate client models using FedSCS."""
        replies = list(replies)

        if not replies:
            return None, None

        if self._global_state is None:
            raise RuntimeError("Global model state is unavailable.")

        client_states: list[dict[str, torch.Tensor]] = []
        client_ids: list[int] = []

        train_losses: list[float] = []
        train_accuracies: list[float] = []
        train_examples: list[int] = []

        for reply in replies:
            if reply.has_error():
                continue

            arrays = reply.content.get("arrays")

            if arrays is None:
                continue

            state_dict = arrays.to_torch_state_dict()

            client_states.append(
                {
                    key: value.detach().cpu().clone()
                    for key, value in state_dict.items()
                }
            )

            client_ids.append(int(reply.metadata.src_node_id))

            metrics = reply.content.get("metrics")

            if metrics is not None:
                train_losses.append(float(metrics["train-loss"]))
                train_accuracies.append(
                    float(metrics["train-accuracy"])
                )
                train_examples.append(
                    int(metrics["num-examples"])
                )

        if not client_states:
            return None, None

        aggregated_state, weights = self._fedscs_aggregation(
            initial_state=self._global_state,
            client_states=client_states,
            client_ids=client_ids,
            server_round=server_round,
        )

        self._global_state = {
            key: value.detach().cpu().clone()
            for key, value in aggregated_state.items()
        }

        if train_losses:
            avg_train_loss = self._weighted_average(
                train_losses,
                train_examples,
            )
            avg_train_accuracy = self._weighted_average(
                train_accuracies,
                train_examples,
            )
        else:
            avg_train_loss = 0.0
            avg_train_accuracy = 0.0

        metrics = MetricRecord(
            {
                "train-loss": float(avg_train_loss),
                "train-accuracy": float(avg_train_accuracy),
                "train-num-clients": len(train_losses),
                "fedscs-num-clients": len(client_states),
                "fedscs-min-trust": float(min(weights)),
                "fedscs-max-trust": float(max(weights)),
            }
        )

        print(f"\n[SERVER] Round {server_round} Training Results")
        print(f"  Clients: {len(client_states)}")
        print(f"  Average client train loss: {avg_train_loss:.4f}")
        print(
            f"  Average client train accuracy: "
            f"{avg_train_accuracy:.4f}"
        )

        return ArrayRecord(aggregated_state), metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate evaluation metrics across clients."""
        replies = list(replies)

        test_losses: list[float] = []
        test_accuracies: list[float] = []
        test_examples: list[int] = []

        for reply in replies:
            if reply.has_error():
                continue

            metrics = reply.content.get("metrics")

            if metrics is None:
                continue

            test_losses.append(float(metrics["test-loss"]))
            test_accuracies.append(
                float(metrics["test-accuracy"])
            )
            test_examples.append(
                int(metrics["num-examples"])
            )

        if not test_losses:
            return None

        avg_test_loss = self._weighted_average(
            test_losses,
            test_examples,
        )
        avg_test_accuracy = self._weighted_average(
            test_accuracies,
            test_examples,
        )

        print(f"\n[SERVER] Round {server_round} Test Results")
        print(f"  Evaluated clients: {len(test_losses)}")
        print(f"  Average test loss: {avg_test_loss:.4f}")
        print(
            f"  Average test accuracy: "
            f"{avg_test_accuracy:.4f}"
        )

        return MetricRecord(
            {
                "test-loss": float(avg_test_loss),
                "test-accuracy": float(avg_test_accuracy),
                "test-num-clients": len(test_losses),
            }
        )

    def summary(self) -> None:
        """Log the FedSCS strategy configuration."""
        print("FedSCS Strategy")
        print(f"  Fraction train: {self.fraction_train}")
        print(f"  Fraction evaluate: {self.fraction_evaluate}")
        print(f"  Minimum train nodes: {self.min_train_nodes}")
        print(
            f"  Minimum evaluate nodes: "
            f"{self.min_evaluate_nodes}"
        )
        print(
            f"  Minimum available nodes: "
            f"{self.min_available_nodes}"
        )
        print(f"  Epsilon: {self.epsilon}")
        print(f"  Beta: {self.beta}")
        print("  Aggregation: FedSCS")
        print("  Peer reference: Leave-one-out peer sum")
        print("  Similarity: Non-negative cosine similarity")
        print("  Temporal smoothing: Enabled")
        print("  Temporal variation correction: Enabled")
