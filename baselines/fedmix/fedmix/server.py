"""..."""
from collections import OrderedDict

import torch
from hydra.utils import instantiate


def gen_evaluate_fn(testloader, device, model):
    """..."""

    def evaluate(server_round, parameters_ndarrays, config):
        """..."""
        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device=device)
        return loss, {"accuracy": accuracy}

    return evaluate


def test(net, dataloader, device):
    """..."""
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    if len(dataloader) == 0:
        raise ValueError("Dataloader can't be 0, exiting...")
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            loss += criterion(output, labels).item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        accuracy = correct / total
        loss /= total
    return float(loss), accuracy
