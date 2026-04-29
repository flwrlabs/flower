"""ClientApp for the split learning demo."""

from __future__ import annotations

from typing import Tuple

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays

from split_learning.task import ClientNet, load_data


class SplitClient(NumPyClient):
    """Client-side participant holding the lower part of the network."""

    def __init__(
        self,
        trainloader,
        testloader,
        model: ClientNet,
        batches_per_round: int,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model.to(device)
        self.batches_per_round = max(1, int(batches_per_round))
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self._cached_batches: list[Tuple[torch.Tensor, torch.Tensor]] = []

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Run the cut-forward step: cache data and return activations/labels."""
        num_batches = int(config.get("batches-per-round", self.batches_per_round))
        activations, labels = self._gather_batches(num_batches)
        if not activations:
            return ndarrays_to_parameters([]), 0, {}

        embedding_tensor = torch.cat(activations).cpu()
        label_tensor = torch.cat(labels).cpu()
        packed = ndarrays_to_parameters(
            [
                embedding_tensor.numpy(),
                label_tensor.numpy(),
            ]
        )
        return packed, len(label_tensor), {}

    def evaluate(self, parameters, config):
        """Apply the activation gradients received from the server."""
        gradients = parameters_to_ndarrays(parameters)
        if not gradients or not self._cached_batches:
            self._cached_batches.clear()
            return 0.0, 0, {}

        grad_tensor = torch.tensor(gradients[0], device=self.device, dtype=torch.float32)
        offset = 0
        self.optimizer.zero_grad()
        for images, _labels in self._cached_batches:
            images = images.to(self.device)
            activations = self.model(images)
            batch_size = activations.shape[0]
            grad_slice = grad_tensor[offset : offset + batch_size]
            activations.backward(grad_slice)
            offset += batch_size
        self.optimizer.step()
        self._cached_batches.clear()
        return 0.0, int(offset), {}

    def _gather_batches(self, num_batches: int) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        self.model.train()
        self._cached_batches.clear()
        activations: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        loader_iter = iter(self.trainloader)
        for _ in range(num_batches):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.trainloader)
                batch = next(loader_iter)

            images, targets = self._unpack_batch(batch)
            images = images.to(self.device)
            targets = targets.to(self.device)
            self._cached_batches.append((images.detach(), targets.detach()))
            with torch.no_grad():
                activations.append(self.model(images).detach())
            labels.append(targets.detach())
        return activations, labels

    @staticmethod
    def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            images = batch.get("image", batch.get("img"))
            labels = batch.get("label")
            return images, labels
        return batch


def client_fn(context: Context):
    """Construct a SplitClient per SuperNode."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    run_config = context.run_config
    batch_size = int(run_config.get("batch-size", 32))
    batches_per_round = int(run_config.get("batches-per-round", 1))
    embedding_size = int(run_config.get("embedding-size", 128))
    learning_rate = float(run_config.get("client-learning-rate", 0.1))

    trainloader, testloader = load_data(partition_id, num_partitions, batch_size)
    model = ClientNet(embedding_size=embedding_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return SplitClient(
        trainloader=trainloader,
        testloader=testloader,
        model=model,
        batches_per_round=batches_per_round,
        learning_rate=learning_rate,
        device=device,
    ).to_client()


app = ClientApp(client_fn=client_fn)
