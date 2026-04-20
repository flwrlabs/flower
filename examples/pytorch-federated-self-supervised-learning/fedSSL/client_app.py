import torch
from torch.utils.data import DataLoader

from flwr.common import Context
from flwr.client import ClientApp, NumPyClient

from fedSSL.model import SimClr, NtxentLoss, get_parameters, set_parameters
from fedSSL.utils import train, load_data


class CifarClient(NumPyClient):
    def __init__(self, partition_id, trainloader, train_epochs, learning_rate):
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.simclr = SimClr()
        self.criterion = NtxentLoss(device=self.device)
        self.optimizer = torch.optim.Adam(self.simclr.parameters(), lr=learning_rate)
        self.train_epochs = train_epochs
        self.trainloader = trainloader

    def fit(self, parameters, config):
        set_parameters(self.simclr, parameters)
        self.simclr.setInference(False)

        results = train(self.simclr, self.partition_id, self.trainloader, self.optimizer, self.criterion,
                        self.train_epochs, self.device)
        
        return get_parameters(self.simclr), len(self.trainloader), results


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    train_epochs = context.run_config['local-train-epochs']
    learning_rate = context.run_config['learning-rate']
    batch_size = context.run_config['batch-size']

    trainset = load_data(partition_id, num_partitions)
    trainloader = DataLoader(trainset, batch_size=batch_size)
    
    return CifarClient(partition_id, trainloader, train_epochs, learning_rate).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
