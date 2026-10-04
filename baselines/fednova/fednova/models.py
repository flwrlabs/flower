"""VGG Model Architecture: Adapted from https://github.com/pytorch/vision.git .

Contains Train, Test function definitions for local client training
"""

import math
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from flwr.common.typing import NDArrays
from torch import nn
from torch.optim.optimizer import Optimizer, required

from fednova.utils import comp_accuracy


class VGG(nn.Module):
    """VGG model."""

    def __init__(self):
        super().__init__()
        self.features = make_layers(cfg["A"])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        """Forward pass through the network."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(network_cfg, batch_norm=False):
    """Define the layer configuration of the VGG-16 network."""
    layers = []
    in_channels = 3
    for v in network_cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


# pylint: disable=too-many-locals,too-many-arguments
def train(
    model, optimizer, trainloader, device, epochs, proximal_mu=0.0
) -> Tuple[float, float]:
    """Train the client model for one round of federated learning."""
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    if proximal_mu > 0.0:
        global_params = [val.detach().clone() for val in model.parameters()]
    else:
        global_params = None
    model.train()

    train_losses = []
    train_accuracy = []
    for _epoch in range(epochs):
        for _batch_idx, (data, target) in enumerate(trainloader):
            # data loading
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # forward pass
            output = model(data)

            if global_params is None:
                loss = criterion(output, target)
            else:
                # Proximal updates for FedProx
                proximal_term = 0.0
                for local_weights, global_weights in zip(
                    model.parameters(), global_params
                ):
                    proximal_term += torch.square(
                        (local_weights - global_weights).norm(2)
                    )
                loss = criterion(output, target) + (proximal_mu / 2) * proximal_term

            # backward pass
            loss.backward()

            # gradient step
            optimizer.step()

            # write log files
            acc = comp_accuracy(output, target)

            train_losses.append(loss.item())
            train_accuracy.append(acc[0].item())

    train_loss = sum(train_losses) / len(train_losses)
    train_acc = sum(train_accuracy) / len(train_accuracy)

    return train_loss, train_acc


def test(model, test_loader, device, *args) -> Tuple[float, Dict[str, float]]:
    """Evaluate the federated model on a test set.

    The server Strategy(FedNova, FedAvg, FedProx) uses the same method to compute
    centralized evaluation on test set using the args. args[0]: int = server round
    args[1]: List[NDArray] = server model parameters args[2]: Dict = {}
    """
    criterion = nn.CrossEntropyLoss()
    if len(args) > 1:
        # load the model parameters
        params_dict = zip(model.state_dict().keys(), args[1])
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    accuracy = []
    total_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            total_loss += criterion(outputs, target).item()
            acc1 = comp_accuracy(outputs, target)
            accuracy.append(acc1[0].item())

    total_loss /= len(test_loader)
    return total_loss, {"accuracy": sum(accuracy) / len(accuracy)}


class ProxSGD(Optimizer):  # pylint: disable=too-many-instance-attributes
    """Optimizer class for FedNova that supports Proximal, SGD, and Momentum updates.

    SGD optimizer modified with support for :
    1. Maintaining a Global momentum buffer, set using : (self.gmf)
    2. Proximal SGD updates, set using : (self.mu)
    Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                    parameter groups
            ratio (float): relative sample size of client
            gmf (float): global/server/slow momentum factor
            mu (float): parameter for proximal local SGD
            lr (float): learning rate
            momentum (float, optional): momentum factor (default: 0)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            dampening (float, optional): dampening for momentum (default: 0)
            nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        params,
        ratio: float,
        gmf=0,
        mu=0,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        variance=0,
    ):
        self.gmf = gmf
        self.ratio = ratio
        self.momentum = momentum
        self.mu = mu
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0
        self.lr = lr

        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "variance": variance,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """Set the optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):  # pylint: disable=too-many-branches
        """Perform a single optimization step."""
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                param_state = self.state[p]

                # if 'old_init' not in param_state:
                # 	param_state['old_init'] = torch.clone(p.data).detach()

                local_lr = group["lr"]

                # apply momentum updates
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply proximal updates
                if self.mu != 0:
                    if param_state["old_init"].device != p.device:
                        param_state["old_init"] = param_state["old_init"].to(p.device)
                    d_p.add_(p.data - param_state["old_init"], alpha=self.mu)

                # update accumalated local updates
                if "cum_grad" not in param_state:
                    param_state["cum_grad"] = torch.clone(d_p).detach()
                    param_state["cum_grad"].mul_(local_lr)

                else:
                    param_state["cum_grad"].add_(d_p, alpha=local_lr)

                p.data.add_(d_p, alpha=-local_lr)

        # compute local normalizing vector a_i
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        etamu = local_lr * self.mu
        if etamu != 0:
            self.local_normalizing_vec *= 1 - etamu
            self.local_normalizing_vec += 1

        if self.momentum == 0 and etamu == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1

    def get_gradient_scaling(self) -> Dict[str, float]:
        """Compute the scaling factor for local client gradients.

        Returns: A dictionary containing weight, tau, and local_norm.
        """
        if self.mu != 0:
            local_tau = torch.tensor(self.local_steps * self.ratio)
        else:
            local_tau = torch.tensor(self.local_normalizing_vec * self.ratio)
        local_stats = {
            "weight": self.ratio,
            "tau": local_tau.item(),
            "local_norm": self.local_normalizing_vec,
        }

        return local_stats

    def set_model_params(self, init_params: NDArrays):
        """Set the model parameters to the given values."""
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_tensor = torch.tensor(init_params[i])
                p.data.copy_(param_tensor)
                param_state["old_init"] = param_tensor
                i += 1

    def set_lr(self, lr: float):
        """Set the learning rate to the given value."""
        for param_group in self.param_groups:
            param_group["lr"] = lr
