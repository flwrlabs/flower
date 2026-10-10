"""Custom Flower Strategy to orchestrate split learning."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy

from split_learning.task import model_to_parameters


class SplitLearningStrategy(Strategy):
    """Strategy that drives cut-forward/cut-backward rounds."""

    def __init__(
        self,
        server_model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        *,
        clients_per_round: int,
        min_available_clients: int = 1,
        accept_failures: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.server_model = server_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.clients_per_round = clients_per_round
        self.min_available_clients = min_available_clients
        self.accept_failures = accept_failures
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.server_model.to(self.device)

        self._cached_grads: dict[str, Parameters] = {}
        self._initial_parameters = model_to_parameters(self.server_model)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self._initial_parameters

    def _select_clients(self, client_manager: ClientManager) -> list[ClientProxy]:
        num_clients = min(self.clients_per_round, client_manager.num_available())
        return client_manager.sample(
            num_clients=num_clients,
            min_num_clients=self.min_available_clients,
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        fit_ins = FitIns(parameters=model_to_parameters(self.server_model), config={})
        return [(client, fit_ins) for client in self._select_clients(client_manager)]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not self.accept_failures and failures:
            return None, {}

        if not results:
            return model_to_parameters(self.server_model), {}

        embeddings, labels, client_lengths, client_ids = self._prepare_batches(results)
        if not embeddings:
            return model_to_parameters(self.server_model), {}

        stacked_embeddings = torch.cat(embeddings, dim=0).requires_grad_(True)
        stacked_embeddings.retain_grad()
        stacked_labels = torch.cat(labels, dim=0)

        self.server_model.train()
        self.optimizer.zero_grad()
        logits = self.server_model(stacked_embeddings)
        loss = self.loss_fn(logits, stacked_labels)
        loss.backward()
        self.optimizer.step()

        grads = stacked_embeddings.grad.detach().cpu().split(client_lengths)
        self._cached_grads = {
            cid: ndarrays_to_parameters([grad.numpy()])
            for cid, grad in zip(client_ids, grads, strict=True)
        }

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == stacked_labels).float().mean().item()

        metrics = {"server_loss": loss.item(), "server_accuracy": accuracy}
        return model_to_parameters(self.server_model), metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        if not self._cached_grads:
            return []

        available = client_manager.all()
        evaluate_ins: list[tuple[ClientProxy, EvaluateIns]] = []
        for cid, grad_params in list(self._cached_grads.items()):
            client = available.get(cid)
            if client:
                evaluate_ins.append(
                    (client, EvaluateIns(grad_params, {"server_round": server_round}))
                )
        self._cached_grads.clear()
        return evaluate_ins

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        if not self.accept_failures and failures:
            return None, {}

        total_examples = sum(res.num_examples for _, res in results)
        if total_examples == 0:
            return None, {}

        loss = sum(res.loss * res.num_examples for _, res in results) / total_examples
        return loss, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        return None

    def _prepare_batches(
        self, results: list[tuple[ClientProxy, FitRes]]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[str]]:
        embeddings: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        lengths: list[int] = []
        client_ids: list[str] = []
        for client, fit_res in results:
            arrays: NDArrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) != 2:
                continue
            activation, lbls = arrays
            activation_tensor = (
                torch.tensor(activation, device=self.device, dtype=torch.float32)
                .detach()
                .requires_grad_(True)
            )
            label_tensor = torch.tensor(lbls, device=self.device, dtype=torch.long)
            embeddings.append(activation_tensor)
            labels.append(label_tensor)
            lengths.append(len(label_tensor))
            client_ids.append(client.cid)
        return embeddings, labels, lengths, client_ids
