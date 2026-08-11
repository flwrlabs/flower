import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from fastai.vision.all import *

from flwr.app import Context
from flwr.client import NumPyClient, start_client
from flwr.clientapp import ClientApp

warnings.filterwarnings("ignore", category=UserWarning)

# Download MNIST dataset
path = untar_data(URLs.MNIST)

# Load dataset
dls = ImageDataLoaders.from_folder(
    path, valid_pct=0.5, train="training", valid="testing", num_workers=0
)

subset_size = 100  # Or whatever
selected_train = np.random.choice(dls.train_ds.items, subset_size, replace=False)
selected_valid = np.random.choice(dls.valid_ds.items, subset_size, replace=False)
# Swap in the subset for the whole thing (Note: this mutates dls, so re-initialize before full training!)
dls.train = dls.test_dl(selected_train, with_labels=True)
dls.valid = dls.test_dl(selected_valid, with_labels=True)

# Define a small model. The E2E test exercises the FastAI/Flower integration,
# so a production-sized pretrained model only adds CPU time.
learn = Learner(
    dls,
    nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10)),
    metrics=error_rate,
)


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in learn.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(learn.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        learn.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        learn.fit(1)
        return self.get_parameters(config={}), len(dls.train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, error_rate = learn.validate()
        return loss, len(dls.valid), {"accuracy": 1 - error_rate}


def client_fn(context: Context):
    return FlowerClient().to_client()


app = ClientApp(
    client_fn=client_fn,
)


if __name__ == "__main__":
    # Start Flower client
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
