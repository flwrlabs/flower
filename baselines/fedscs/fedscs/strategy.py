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
    ) -> None:
        """Initialize FedSCS strategy."""

        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_nodes = min_train_nodes
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.epsilon = epsilon

        # Used ONLY for Flower's node sampling/configuration.
        # Model aggregation itself is implemented by FedSCS below.
        self._sampling_strategy = FedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
        )

        # Historical FedSCS state for each client.
        self.client_histories: dict[int, dict[str, list[float]]] = {}

        # Global model before local training.
        self._global_state: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _flatten_update(
        update: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Flatten a model update into one vector."""

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
        """Compute non-negative cosine similarity."""

        u_norm = torch.linalg.vector_norm(u)
        v_norm = torch.linalg.vector_norm(v)

        if u_norm.item() <= 0.0 or v_norm.item() <= 0.0:
            return 0.0

        similarity = F.cosine_similarity(
            u.unsqueeze(0),
            v.unsqueeze(0),
            dim=1,
        ).item()

        return max(float(similarity), 0.0)

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure all clients for local training."""

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
        """Configure all clients for evaluation."""

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

        if num_clients == 1:
            return client_states[0], [1.0]

        # ---------------------------------------------------------------
        # Step 1: Compute client updates
        #
        # Delta_i = W_i - W_global
        # ---------------------------------------------------------------
        delta_states = []

        for client_state in client_states:
            delta = {
                key: (
                    client_state[key].detach().cpu().float()
                    - initial_state[key].detach().cpu().float()
                )
                for key in initial_state
            }
            delta_states.append(delta)

        # ---------------------------------------------------------------
        # Step 2: Flatten updates
        # ---------------------------------------------------------------
        flat_updates = [
            self._flatten_update(delta)
            for delta in delta_states
        ]

        # ---------------------------------------------------------------
        # Step 3: Leave-one-out peer reference
        #
        # P_i = sum_{j != i} Delta_j
        # ---------------------------------------------------------------
        total_update = torch.stack(flat_updates).sum(dim=0)

        peer_sums = [
            total_update - flat_updates[i]
            for i in range(num_clients)
        ]

        # ---------------------------------------------------------------
        # Step 4: Stable cosine similarity
        #
        # rho_i = max(cosine(Delta_i, P_i), 0)
        # ---------------------------------------------------------------
        rho = [
            self._cosine_similarity(
                flat_updates[i],
                peer_sums[i],
            )
            for i in range(num_clients)
        ]

        # ---------------------------------------------------------------
        # Step 5: Temporal trust score
        # ---------------------------------------------------------------
        trust_scores = []

        for i, client_id in enumerate(client_ids):
            history = self.client_histories.setdefault(
                client_id,
                {
                    "s": [],
                    "rho": [],
                    "nu": [],
                    "s_tilde": [],
                    "a": [],
                },
            )

            previous_s = (
                history["s"][-1]
                if history["s"]
                else 1.0
            )

            t = max(server_round, 1)

            s_i = (
                ((t - 1) / t) * previous_s
                + (1.0 / t) * rho[i]
            )

            history["s"].append(s_i)
            history["rho"].append(rho[i])

            trust_scores.append(s_i)

        # ---------------------------------------------------------------
        # Step 6: Temporal variation correction
        # ---------------------------------------------------------------
        corrected_scores = []

        for client_id, s_i in zip(client_ids, trust_scores):
            history = self.client_histories[client_id]

            if len(history["s"]) > 1:
                previous_s = history["s"][-2]
            else:
                previous_s = 1.0

            nu_i = abs(s_i - previous_s) / (
                previous_s + self.epsilon
            )

            s_tilde_i = s_i / (1.0 + nu_i)

            history["nu"].append(nu_i)
            history["s_tilde"].append(s_tilde_i)

            corrected_scores.append(s_tilde_i)

        # ---------------------------------------------------------------
        # Step 7: Normalize FedSCS trust scores
        # ---------------------------------------------------------------
        total_score = sum(corrected_scores)

        if total_score <= self.epsilon:
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
        # Step 8: FedSCS model aggregation
        #
        # W_(t+1) = sum_i a_i W_i
        # ---------------------------------------------------------------
        aggregated_state = {}

        for key in initial_state:
            aggregated_state[key] = sum(
                aggregation_weights[i]
                * client_states[i][key].detach().cpu().float()
                for i in range(num_clients)
            )

        # Preserve integer/buffer dtypes.
        for key, initial_value in initial_state.items():
            if not torch.is_floating_point(initial_value):
                aggregated_state[key] = aggregated_state[key].to(
                    dtype=initial_value.dtype
                )

        print(f"\n[FedSCS] Round {server_round}")
        print(f"  Participating clients: {num_clients}")

        for client_id, rho_i, weight in zip(
            client_ids,
            rho,
            aggregation_weights,
        ):
            print(
                f"  Client {client_id}: "
                f"rho={rho_i:.4f}, "
                f"trust_weight={weight:.4f}"
            )

        return aggregated_state, aggregation_weights

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate models with FedSCS and average training metrics."""

        replies = list(replies)

        if not replies:
            return None, None

        if self._global_state is None:
            raise RuntimeError(
                "Global model state is unavailable."
            )

        client_states = []
        client_ids = []

        train_losses = []
        train_accuracies = []
        train_examples = []

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

            client_ids.append(
                int(reply.metadata.src_node_id)
            )

            # -----------------------------------------------------------
            # Read client training metrics
            # -----------------------------------------------------------
            metrics = reply.content.get("metrics")

            if metrics is not None:
                train_losses.append(
                    float(metrics["train-loss"])
                )
                train_accuracies.append(
                    float(metrics["train-accuracy"])
                )
                train_examples.append(
                    int(metrics["num-examples"])
                )

        if not client_states:
            return None, None

        # ---------------------------------------------------------------
        # ACTUAL FedSCS MODEL AGGREGATION
        # ---------------------------------------------------------------
        aggregated_state, weights = self._fedscs_aggregation(
            initial_state=self._global_state,
            client_states=client_states,
            client_ids=client_ids,
            server_round=server_round,
        )

        # Store global model.
        self._global_state = {
            key: value.detach().cpu().clone()
            for key, value in aggregated_state.items()
        }

        # ---------------------------------------------------------------
        # Average client training metrics
        # ---------------------------------------------------------------
        if train_losses:
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_accuracy = (
                sum(train_accuracies) / len(train_accuracies)
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
                "fedscs-mean-trust": float(
                    sum(weights) / len(weights)
                ),
            }
        )

        print(
            f"\n[SERVER] Round {server_round} Training Results"
        )
        print(
            f"  Clients: {len(train_losses)}"
        )
        print(
            f"  Average client train loss: "
            f"{avg_train_loss:.4f}"
        )
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
        """Average test loss and accuracy across all clients."""

        replies = list(replies)

        test_losses = []
        test_accuracies = []

        for reply in replies:
            if reply.has_error():
                continue

            metrics = reply.content.get("metrics")

            if metrics is None:
                continue

            test_losses.append(
                float(metrics["test-loss"])
            )
            test_accuracies.append(
                float(metrics["test-accuracy"])
            )

        if not test_losses:
            return None

        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_test_accuracy = (
            sum(test_accuracies) / len(test_accuracies)
        )

        print(
            f"\n[SERVER] Round {server_round} Test Results"
        )
        print(
            f"  Evaluated clients: {len(test_losses)}"
        )
        print(
            f"  Average test loss: "
            f"{avg_test_loss:.4f}"
        )
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
        print(
            f"  Fraction train: "
            f"{self.fraction_train}"
        )
        print(
            f"  Fraction evaluate: "
            f"{self.fraction_evaluate}"
        )
        print(
            f"  Minimum train nodes: "
            f"{self.min_train_nodes}"
        )
        print(
            f"  Minimum evaluate nodes: "
            f"{self.min_evaluate_nodes}"
        )
        print(
            f"  Minimum available nodes: "
            f"{self.min_available_nodes}"
        )
        print(
            f"  Epsilon: {self.epsilon}"
        )
        print("  Aggregation: FedSCS")
        print(
            "  Peer reference: Leave-one-out peer sum"
        )
        print(
            "  Similarity: Non-negative cosine similarity"
        )
        print(
            "  Temporal smoothing: Enabled"
        )
        print(
            "  Temporal variation correction: Enabled"
        )
