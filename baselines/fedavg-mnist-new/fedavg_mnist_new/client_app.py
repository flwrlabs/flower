# fedavg_mnist_new/client.py
import torch
from fedavg_mnist_new.model import MnistCNN, train_local, test
import flwr as fl

class MnistClient(fl.client.NumPyClient):
    def __init__(self, cid, loader, device, local_epochs=5, lr=0.1):
        self.cid = cid
        self.loader = loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.model = MnistCNN().to(device)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        train_local(self.model, self.loader, epochs=self.local_epochs, device=self.device, lr=self.lr)
        new_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return new_params, len(self.loader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.loader, self.device)
        return float(loss), len(self.loader.dataset), {"accuracy": float(accuracy)}

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def to_client(self):
        """Convert this NumPyClient to a standard Client (to avoid deprecation warnings)."""
        return fl.client.cast(self)
