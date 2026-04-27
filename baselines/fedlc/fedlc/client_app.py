"""fedlc: A Flower Baseline."""

import torch

from fedlc.dataset import load_data
from fedlc.model import CNNModel, get_parameters, set_parameters, train
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


class LogitCorrectedLoss(torch.nn.CrossEntropyLoss):
    def __init__(
        self,
        num_classes,
        labels,
        tau,
        device,
    ):
        super().__init__()
        class_count = torch.zeros(num_classes).long()
        labels, counts = labels.unique(
            sorted=True, return_counts=True, return_inverse=False
        )
        class_count[labels] = counts
        class_count = class_count.to(device)
        self.correction = tau * class_count.pow(-0.25)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        corrected_logits = logits - self.correction
        return super().forward(corrected_logits, target)

class RestrictedSoftmaxLoss(torch.nn.CrossEntropyLoss):
    def __init__(
        self,
        num_classes,
        labels,
        alpha,
        device,
    ):
        super().__init__()
        class_count = torch.ones(num_classes) * alpha
        labels = labels.unique(sorted=True, return_counts=False, return_inverse=False)
        for c in labels:
            class_count[c] = 1.0
        class_count = class_count.unsqueeze(dim=0).to(device)
        self.correction = class_count

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        corrected_logits = logits * self.correction
        return super().forward(corrected_logits, target)


class FlowerClient(NumPyClient):
    def __init__(
        self, 
        net,
        trainloader,
        labels,
        context,
    ):
        self.net = net
        self.trainloader = trainloader

        local_epochs = int(context.run_config["local-epochs"])
        learning_rate = float(context.run_config["learning-rate"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alg = str(context.run_config["alg"])
        proximal_mu, alpha, tau = 0, 0, 0
        if alg == "fedavg":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif alg == "fedprox":
            proximal_mu = float(context.run_config["proximal-mu"])
            self.criterion = torch.nn.CrossEntropyLoss()
        elif alg == "fedrs":
            alpha = float(context.run_config["fedrs-alpha"])
            self.criterion = RestrictedSoftmaxLoss(
                net.fc.out_features,  # num_classes
                labels,
                alpha,
                device
            )
        elif alg == "fedlc":
            tau = float(context.run_config["tau"])
            self.criterion = LogitCorrectedLoss(
                net.fc.out_features,  # num_classes
                labels,
                tau,
                device,
            )
        else:
            raise ValueError(f"alg={alg} not supported!")
        
        self.proximal_mu = proximal_mu
        self.alpha = alpha
        self.tau = tau

        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device

    def get_parameters(self, config):
        """Return the parameters of the current net."""
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            self.learning_rate,
            self.criterion,
            self.proximal_mu,
        )
        return (
            get_parameters(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    trainloader, labels, num_classes = load_data(context)
    net = CNNModel(num_classes)

    return FlowerClient(
        net, trainloader, labels, context
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
