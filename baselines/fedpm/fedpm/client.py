"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from copy import deepcopy
from logging import INFO
from typing import Dict, List

import flwr
import numpy as np
import torch
import torch.nn as nn
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy.stats import bernoulli
from torch.utils.data import DataLoader, dataset

# TO CHECK
from fedpm.models import get_parameters, load_model, set_parameters


class FedPMClient(flwr.client.Client):
    def __init__(
        self,
        model_cfg: DictConfig,
        client_id: int,
        train_data_loader: DataLoader,
        test_data_loader: DataLoader,
    ) -> None:
        self.model_cfg = model_cfg
        self.client_id = client_id
        self.train_data_loader = train_data_loader
        # Device is set based on the `client_resources` passed to start_simulation
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_data_loader = test_data_loader
        self.local_model = load_model(self.model_cfg).to(self.device)
        self.epsilon = 0.01

    def sample_mask(self, mask_probs: Dict) -> List[np.ndarray]:
        sampled_mask = []
        with torch.no_grad():
            for layer_name, layer in mask_probs.items():
                if "mask" in layer_name:
                    theta = torch.sigmoid(layer).cpu().numpy()
                    updates_s = bernoulli.rvs(theta)
                    updates_s = np.where(updates_s == 0, self.epsilon, updates_s)
                    updates_s = np.where(updates_s == 1, 1 - self.epsilon, updates_s)
                    sampled_mask.append(updates_s)
                else:
                    sampled_mask.append(layer.cpu().numpy())

        return sampled_mask

    def fit(self, fitins: FitIns) -> FitRes:
        self.set_parameters(fitins.parameters)

        config = fitins.config

        local_epochs = config["local_epochs"]
        local_lr = config["local_lr"]

        mask_probs = self.train_fedpm(
            model=self.local_model,
            trainloader=self.train_data_loader,
            local_epochs=local_epochs,
            local_lr=local_lr,
        )
        sampled_mask = self.sample_mask(mask_probs)
        parameters = ndarrays_to_parameters(sampled_mask)
        status = Status(code=Code.OK, message="Success")

        return FitRes(
            status=status,
            parameters=parameters,
            num_examples=len(self.train_data_loader),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        loss_fn = torch.nn.CrossEntropyLoss()
        self.set_parameters(ins.parameters)
        loss, accuracy = 0, 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _batch_id, (inputs, labels) in enumerate(self.test_data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss += loss_fn(outputs, labels)
                correct += (predicted == labels).sum().item()
                total += labels.shape[0]
            accuracy = 100 * correct / total
        status = Status(Code.OK, message="Success")
        log(INFO, "Client %s Accuracy: %f   Loss: %f", self.client_id, accuracy, loss)
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.test_data_loader),
            metrics={"accuracy": float(accuracy), "loss": float(loss)},
        )

    def train_fedpm(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        local_epochs: int,
        local_lr: float,
        loss_fn=nn.CrossEntropyLoss(reduction="mean"),
        optimizer: torch.optim.Optimizer = None,
    ) -> Dict:
        """Compute local epochs, the training strategies depends on the adopted
        model.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=local_lr)

        for _epoch in range(local_epochs):
            running_loss = 0
            total = 0
            correct = 0
            for _batch_idx, (train_x, train_y) in enumerate(trainloader):
                train_x, train_y = train_x.to(self.device), train_y.to(self.device)
                total += train_x.size(0)
                optimizer.zero_grad()
                y_pred = model(train_x)
                loss = loss_fn(y_pred, train_y)
                running_loss += loss.item()
                _, pred_y = torch.max(y_pred.data, 1)
                correct += (pred_y == train_y).sum().item()
                loss.backward()
                optimizer.step()

        return model.state_dict()

    def get_parameters(self, ins: GetParametersIns = None) -> GetParametersRes:
        ndarrays = get_parameters(self.local_model)
        parameters = ndarrays_to_parameters(ndarrays)
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(status=status, parameters=parameters)

    def set_parameters(self, parameters):
        set_parameters(self.local_model, parameters_to_ndarrays(parameters))


class DenseClient(flwr.client.Client):
    def __init__(
        self,
        model_cfg: DictConfig,
        compressor_cfg: DictConfig,
        client_id: int,
        train_data_loader: DataLoader,
        test_data_loader: DataLoader,
    ) -> None:
        self.client_id = client_id
        self.compressor_cfg = compressor_cfg
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        # Device is set based on the `client_resources` passed to start_simulation
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_model = load_model(model_cfg).to(self.device)
        self.compressor = None
        self.compression = compressor_cfg.compress
        if self.compression:
            # TODO: make this better in 2nd round of Hydra fixes
            self.compressor = instantiate(
                compressor_cfg[compressor_cfg.type], device=self.device
            )

    def fit(self, fitins: FitIns) -> FitRes:
        self.set_parameters(fitins.parameters)
        deltas = self.train_dense(
            model=self.local_model,
            trainloader=self.train_data_loader,
        )

        if self.compression:
            compressed_delta, avg_bitrate = self.compressor.compress(
                updates=deltas,
            )
            round_rate = avg_bitrate
        else:
            compressed_delta = []
            for _i, (_name, param) in enumerate(deltas.items()):
                compressed_delta.append(param.cpu().numpy())
            round_rate = 32
        parameters = ndarrays_to_parameters(compressed_delta)
        status = Status(code=Code.OK, message="Success")
        metrics = {
            "Rate": np.mean(round_rate),
        }
        return FitRes(
            status=status,
            parameters=parameters,
            num_examples=len(self.train_data_loader),
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        loss_fn = torch.nn.CrossEntropyLoss()
        self.set_parameters(ins.parameters)
        loss, accuracy = 0, 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _batch_id, (inputs, labels) in enumerate(self.test_data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss += loss_fn(outputs, labels)
                correct += (predicted == labels).sum().item()
                total += labels.shape[0]
            accuracy = 100 * correct / total
        status = Status(Code.OK, message="Success")
        log(INFO, "Client %s Accuracy: %f   Loss: %f", self.client_id, accuracy, loss)
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.test_data_loader),
            metrics={"accuracy": float(accuracy), "loss": float(loss)},
        )

    def train_dense(
        self,
        model: nn.Module,
        trainloader: dataset,
        loss_fn=nn.CrossEntropyLoss(reduction="mean"),
        optimizer: torch.optim.Optimizer = None,
    ):
        """Train the network on the training set."""
        if self.compressor_cfg.compress:
            # TODO: make this better in 2nd round of Hydra fixes
            if self.compressor_cfg.type == "sign_sgd":
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=self.compressor.local_lr
                )
            if self.compressor_cfg.type == "qsgd":
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=self.compressor.local_lr
                )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.compressor.local_lr
            )

        global_model = deepcopy(model.state_dict())
        model.train()
        for _epoch in range(self.compressor_cfg.local_epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(trainloader.dataset)

        full_grad = {}
        with torch.no_grad():
            for _k, (name, param) in enumerate(model.state_dict().items()):
                full_grad[name] = param - global_model[name]
        return full_grad

    def get_parameters(self, ins: GetParametersIns = None) -> GetParametersRes:
        ndarrays = get_parameters(self.local_model)
        parameters = ndarrays_to_parameters(ndarrays)
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(status=status, parameters=parameters)

    def set_parameters(self, parameters):
        set_parameters(self.local_model, parameters_to_ndarrays(parameters))
