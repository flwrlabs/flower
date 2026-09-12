# fedavg_mnist_new/dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset, ConcatDataset

def load_datasets():
    """Load MNIST training and test datasets."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def partition_dataset(dataset, num_clients=10, iid=True, seed=42):
    """Split a dataset into `num_clients` subsets (IID or non-IID)."""
    if iid:
        num_items = len(dataset) // num_clients
        lengths = [num_items] * num_clients
        return random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed))
    else:
        targets = torch.tensor(dataset.targets)  # labels
        indices = targets.argsort()
        sorted_ds = Subset(dataset, indices)
        shard_size = len(sorted_ds) // (num_clients * 2)
        shards = []
        for i in range(num_clients * 2):
            shard_indices = list(range(i * shard_size, (i + 1) * shard_size))
            shards.append(Subset(sorted_ds, shard_indices))
        rng = torch.Generator().manual_seed(seed)
        shard_perm = torch.randperm(num_clients * 2, generator=rng)
        clients = []
        for i in range(num_clients):
            shard1 = shards[shard_perm[2 * i]]
            shard2 = shards[shard_perm[2 * i + 1]]
            clients.append(ConcatDataset([shard1, shard2]))
        return clients

def create_client_loaders(client_datasets, batch_size=32, shuffle=True):
    """Create a DataLoader for each client dataset."""
    from torch.utils.data import DataLoader
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=shuffle) for ds in client_datasets]
    return client_loaders
