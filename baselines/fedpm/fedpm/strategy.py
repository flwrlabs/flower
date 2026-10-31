"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

from typing import Dict, List, Optional, Tuple, Union

import flwr
import numpy as np
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedpm.models import get_parameters, load_model, set_parameters


class FedPMStrategy(flwr.server.strategy.Strategy):
    def __init__(
        self,
        model_cfg: DictConfig,
        global_data_loader: DataLoader,
        loss_fn,
        device: str,
        local_epochs: int,
        local_lr: float,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.model_cfg = model_cfg
        self.global_dataset = global_data_loader
        self.loss_fn = loss_fn
        self.device = device
        self.global_model = load_model(self.model_cfg).to(self.device)

        # to send to client as part in configure fit
        self.local_epochs = local_epochs
        self.local_lr = local_lr

        self.alphas = None
        self.betas = None
        self.reset_mask_prob()
        self.masked_layer_id = []
        i = 0
        for name, _layer in self.global_model.named_parameters():
            if "mask" in name:
                self.masked_layer_id.append(i)
            i += 1

    def __repr__(self) -> str:
        return "Federated Probabilistic Mask"

    def initialize_parameters(self, client_manager: ClientManager):
        return flwr.common.ndarrays_to_parameters(get_parameters(self.global_model))

    def reset_mask_prob(self):
        self.alphas = []
        self.betas = []
        with torch.no_grad():
            for _k, val in self.global_model.named_parameters():
                self.alphas.append(np.ones_like(val.cpu().numpy()))
                self.betas.append(np.ones_like(val.cpu().numpy()))

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_configurations = []

        for _idx, client in enumerate(clients):
            fit_configurations.append(
                (
                    client,
                    FitIns(
                        parameters,
                        {
                            "iter_num": server_round,
                            "local_epochs": self.local_epochs,
                            "local_lr": self.local_lr,
                        },
                    ),
                )
            )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.reset_mask_prob()

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        sampled_clients = len(weights_results)
        parameters_aggregated = aggregate(weights_results)

        layer_id = 0
        for layer in parameters_aggregated:
            self.alphas[layer_id] += layer * sampled_clients
            self.betas[layer_id] += sampled_clients - layer * sampled_clients
            layer_id += 1

        # Use Median of Betas
        updated_params = []
        for layer_id in range(len(self.alphas)):
            if layer_id in self.masked_layer_id:
                avg_p = (self.alphas[layer_id] - 1) / (
                    self.alphas[layer_id] + self.betas[layer_id] - 2
                )
                updated_params.append(np.log(avg_p / (1 - avg_p)))
            else:
                updated_params.append(parameters_aggregated[layer_id])

        return ndarrays_to_parameters(updated_params), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        correct = 0
        total = 0
        set_parameters(self.global_model, parameters_to_ndarrays(parameters))
        self.global_model.to(self.device)
        with torch.no_grad():
            inputs, labels = next(iter(self.global_dataset))
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.global_model.zero_grad()
            outputs = self.global_model(inputs)
            loss = self.loss_fn(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        print(f"Round {server_round} - Loss: {loss}  /  Accuracy: {accuracy}")

        return loss, {"accuracy": accuracy}

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


class DenseStrategy(flwr.server.strategy.Strategy):
    def __init__(
        self,
        model_cfg: DictConfig,
        compressor_cfg: DictConfig,
        global_data_loader: DataLoader,
        loss_fn,
        device="cpu",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        sim_folder=None,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.model_cfg = model_cfg
        self.compressor_cfg = compressor_cfg
        self.global_dataset = global_data_loader
        self.loss_fn = instantiate(loss_fn)
        self.device = device

        self.global_model = load_model(self.model_cfg).to(self.device)
        self.sim_folder = sim_folder
        if compressor_cfg.compress:
            if compressor_cfg.type == "sign_sgd":
                self.server_lr = compressor_cfg.sign_sgd.server_lr
            if compressor_cfg.type == "qsgd":
                self.server_lr = compressor_cfg.qsgd.server_lr
        else:
            self.server_lr = compressor_cfg.server_lr

        self.layer_id = []
        i = 0
        for _, _ in self.global_model.named_parameters():
            self.layer_id.append(i)
            i += 1

    def __repr__(self) -> str:
        return "Dense Training"

    def initialize_parameters(self, client_manager: ClientManager):
        return flwr.common.ndarrays_to_parameters(get_parameters(self.global_model))

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_configurations = []
        for _idx, client in enumerate(clients):
            fit_configurations.append(
                (client, FitIns(parameters, {"iter_num": server_round}))
            )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        deltas_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        rate_results = [fit_res.metrics["Rate"] for _, fit_res in results]
        log_metrics = {"Rate": [np.mean(rate_results)]}
        for _, fit_res in results:
            for name, val in fit_res.metrics.items():
                if log_metrics.get("name") is None:
                    log_metrics[name] = [val]
                else:
                    log_metrics[name].append(val)
        for name, val in log_metrics.items():
            log_metrics[name] = np.mean(val)

        deltas_aggregated = aggregate(deltas_results)

        return ndarrays_to_parameters(deltas_aggregated), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        correct = torch.tensor(0, device=self.device)
        total = 0
        set_parameters(self.global_model, parameters_to_ndarrays(parameters))
        # self.set_parameters(parameters)
        with torch.no_grad():
            inputs, labels = next(iter(self.global_dataset))
            self.global_model.zero_grad()
            labels = labels.to(self.device)
            outputs = self.global_model(inputs.to(self.device))
            loss = self.loss_fn(outputs, labels.to(self.device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        print(f"Round {server_round} - Loss: {loss}  /  Accuracy: {accuracy}")
        return loss, {"accuracy": accuracy}

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
