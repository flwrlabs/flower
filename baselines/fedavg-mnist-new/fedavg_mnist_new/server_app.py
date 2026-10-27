# fedavg_mnist_new/server_app.py
import torch
from fedavg_mnist_new.model import MnistCNN, test
from torch.utils.data import DataLoader
from flwr.common import NDArrays

def evaluate_global(weights: NDArrays, test_loader, device):
    model = MnistCNN().to(device)
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    loss, accuracy = test(model, test_loader, device)
    return loss, {"accuracy": accuracy}
