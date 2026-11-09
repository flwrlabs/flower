"""Model definitions for the FedHT baseline.

Three model types covering all experiments: sparse linear regression
(Simulation I), sparse logistic regression (Simulation II), and sparse
softmax regression (MNIST). All are linear with no hidden layers; the
sparsity constraint is enforced externally by the strategy.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SparseLinearRegression(nn.Module):
    """Single linear layer for sparse linear regression (Simulation I)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predictions."""
        return self.linear(x).squeeze(1)


class SparseLogisticRegression(nn.Module):
    """Single linear layer for binary sparse logistic regression (Simulation II).

    Sigmoid is omitted here because BCEWithLogitsLoss combines it with the loss.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw logits."""
        return self.linear(x).squeeze(1)


class SparseSoftmaxRegression(nn.Module):
    """Linear multi-class classifier for sparse softmax regression (MNIST)."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits."""
        return self.linear(x)


def get_weights(net: nn.Module) -> list[np.ndarray]:
    """Extract model parameters as a list of numpy arrays."""
    return [val.cpu().detach().numpy() for val in net.state_dict().values()]


def set_weights(net: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load a list of numpy arrays into the model state dict."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def _train(
    net: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    local_steps: int,
    use_local_ht: bool,
    tau: int,
    y_dtype: torch.dtype | None = None,
) -> None:
    """Shared SGD loop used by all three task-specific train functions."""
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()

    step = 0
    data_iter = iter(loader)
    while step < local_steps:
        try:
            X_batch, y_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            X_batch, y_batch = next(data_iter)

        X_batch = X_batch.to(device)
        y_batch = (
            y_batch.to(device=device, dtype=y_dtype)
            if y_dtype is not None
            else y_batch.to(device)
        )
        optimizer.zero_grad()
        loss = criterion(net(X_batch), y_batch)
        loss.backward()
        optimizer.step()

        if use_local_ht and tau > 0:
            _apply_local_ht(net, tau)

        step += 1


def train_linear(
    net: SparseLinearRegression,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    local_steps: int,
    use_local_ht: bool = False,
    tau: int = 0,
) -> None:
    """Run K local SGD steps for sparse linear regression."""
    _train(net, nn.MSELoss(), loader, device, lr, local_steps, use_local_ht, tau)


def train_logistic(
    net: SparseLogisticRegression,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    local_steps: int,
    use_local_ht: bool = False,
    tau: int = 0,
) -> None:
    """Run K local SGD steps for sparse logistic regression."""
    _train(
        net,
        nn.BCEWithLogitsLoss(),
        loader,
        device,
        lr,
        local_steps,
        use_local_ht,
        tau,
        y_dtype=torch.float32,
    )


def train_softmax(
    net: SparseSoftmaxRegression,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    local_steps: int,
    use_local_ht: bool = False,
    tau: int = 0,
) -> None:
    """Run K local SGD steps for sparse softmax regression."""
    _train(
        net,
        nn.CrossEntropyLoss(),
        loader,
        device,
        lr,
        local_steps,
        use_local_ht,
        tau,
        y_dtype=torch.int64,
    )


def _apply_local_ht(net: nn.Module, tau: int) -> None:
    """Apply hard thresholding in-place to all model parameters."""
    with torch.no_grad():
        params = list(net.parameters())
        sizes = [p.numel() for p in params]
        flat = torch.cat([p.flatten() for p in params])

        if tau >= flat.numel():
            return

        abs_flat = flat.abs()
        cutoff = torch.kthvalue(abs_flat, flat.numel() - tau).values
        mask = abs_flat >= cutoff

        nonzero = mask.sum().item()
        if nonzero > tau:
            excess = int(nonzero - tau)
            indices = mask.nonzero(as_tuple=True)[0]
            mask[indices[:excess]] = False

        flat = flat * mask

        start = 0
        for p, size in zip(params, sizes):
            p.copy_(flat[start : start + size].reshape(p.shape))
            start += size


def eval_linear(
    net: SparseLinearRegression,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, int]:
    """Return mean MSE loss and sample count over the validation set."""
    criterion = nn.MSELoss(reduction="sum")
    net.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            total_loss += criterion(net(X_batch), y_batch).item()
            total_samples += len(y_batch)
    return total_loss / max(total_samples, 1), total_samples


def eval_logistic(
    net: SparseLogisticRegression,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int]:
    """Return mean BCE loss, accuracy, and sample count over the validation set."""
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.float().to(device)
            logits = net(X_batch)
            total_loss += criterion(logits, y_batch).item()
            preds = (logits > 0).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / max(total, 1), correct / max(total, 1), total


def eval_softmax(
    net: SparseSoftmaxRegression,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int]:
    """Return mean cross-entropy loss, accuracy, and sample count."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.long().to(device)
            logits = net(X_batch)
            total_loss += criterion(logits, y_batch).item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / max(total, 1), correct / max(total, 1), total
