from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from split_learning.client_app import SplitClient  # noqa: E402
from split_learning.strategy import SplitLearningStrategy  # noqa: E402
from split_learning.task import ClientNet, ServerNet  # noqa: E402


def _fit_result(cid: str, embeddings: np.ndarray, labels: np.ndarray):
    status = Status(code=Code.OK, message="")
    params = ndarrays_to_parameters([embeddings, labels])
    return SimpleNamespace(cid=cid), FitRes(
        status=status, parameters=params, num_examples=len(labels), metrics={}
    )


class _DictDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


def test_strategy_generates_gradients_per_client():
    torch.manual_seed(0)
    server_model = ServerNet(embedding_size=4, num_classes=2)
    optimizer = torch.optim.SGD(server_model.parameters(), lr=0.1)
    strategy = SplitLearningStrategy(
        server_model=server_model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        clients_per_round=2,
        min_available_clients=1,
        device=torch.device("cpu"),
    )

    initial_first_weight = next(server_model.parameters()).detach().clone()
    client_a = torch.randn(2, 4)
    labels_a = torch.tensor([0, 1])
    client_b = torch.randn(2, 4)
    labels_b = torch.tensor([1, 0])

    results = [
        _fit_result("cid-0", client_a.numpy(), labels_a.numpy()),
        _fit_result("cid-1", client_b.numpy(), labels_b.numpy()),
    ]

    parameters, metrics = strategy.aggregate_fit(server_round=1, results=results, failures=[])

    assert parameters is not None
    assert "server_loss" in metrics and "server_accuracy" in metrics
    assert set(strategy._cached_grads.keys()) == {"cid-0", "cid-1"}
    assert parameters_to_ndarrays(strategy._cached_grads["cid-0"])[0].shape == client_a.shape
    updated_first_weight = next(server_model.parameters()).detach()
    assert not torch.allclose(initial_first_weight, updated_first_weight)


def test_client_applies_server_gradients_and_updates_weights():
    torch.manual_seed(1)
    images = torch.ones(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    dataset = _DictDataset(images, labels)
    trainloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = ClientNet(embedding_size=8)
    client = SplitClient(
        trainloader=trainloader,
        testloader=trainloader,
        model=model,
        batches_per_round=1,
        learning_rate=0.1,
        device=torch.device("cpu"),
    )

    before = [p.detach().clone() for p in client.model.parameters()]
    fit_params, num_examples, _ = client.fit(parameters=None, config={"batches-per-round": 1})
    grads = np.ones_like(parameters_to_ndarrays(fit_params)[0])
    loss, evaluated_examples, _ = client.evaluate(ndarrays_to_parameters([grads]), config={})

    after = [p.detach() for p in client.model.parameters()]
    assert num_examples == evaluated_examples == 2
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert client._cached_batches == []
