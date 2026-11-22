# fedavg_mnist_new/main.py
import torch
import flwr as fl
from flwr.common import NDArrays
from fedavg_mnist_new.dataset import load_datasets, partition_dataset, create_client_loaders
from fedavg_mnist_new.client_app import MnistClient
from fedavg_mnist_new.model import MnistCNN
from fedavg_mnist_new.server_app import evaluate_global
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Hyperparameters and device configuration
NUM_CLIENTS = 10
LOCAL_EPOCHS = 5
LR = 0.1
DEVICE = torch.device("cpu")

def main():
    # 1. Load datasets
    train_dataset, test_dataset = load_datasets()

    # 2. Partition training data among clients (IID)
    client_datasets = partition_dataset(train_dataset, num_clients=NUM_CLIENTS, iid=True)

    # 3. Create DataLoaders for each client
    client_loaders = create_client_loaders(client_datasets, batch_size=32, shuffle=True)

    # 4. Create a DataLoader for the global test set
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 5. Define the global evaluation function for the server
    def evaluate_fn(server_round, parameters, config):
        return evaluate_global(parameters, test_loader, DEVICE)

    # 6. Define a client function using the new Context signature to avoid deprecation warnings
    def client_fn(cid: str) -> fl.client.Client:
        return MnistClient(cid, client_loaders[int(cid)], DEVICE, local_epochs=LOCAL_EPOCHS, lr=LR)


    # 7. Configure the FedAvg strategy with centralized evaluation
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        evaluate_fn=evaluate_fn,
        # Optionally: specify min_fit_clients, etc.
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    # 8. Start the Flower simulation for the specified number of rounds
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    print("Simulation finished. Metrics:")
    print(history.metrics_centralized)

if __name__ == "__main__":
    main()
