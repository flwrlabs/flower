"""Handle basic dataset creation."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision.datasets import MNIST


def load_datasets(  # pylint: disable=too-many-arguments
    dataset_cfg: DictConfig,
    val_ratio: float = 0.1,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    dataset_cfg : DictConfig
        Parameterised the dataset creation process (defined in config file)
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the
        DataLoader for testing.
    """
    datasets, testset = _partition_data(
        dataset_cfg.num_clients, dataset_cfg.iid, dataset_cfg.balance, seed
    )
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(
            DataLoader(ds_train, batch_size=dataset_cfg.batch_size, shuffle=True)
        )
        valloaders.append(DataLoader(ds_val, batch_size=dataset_cfg.batch_size))
    return (
        trainloaders,
        valloaders,
        DataLoader(testset, batch_size=dataset_cfg.batch_size),
    )


def _download_data() -> Tuple[Dataset, Dataset]:
    """Download (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


def _partition_data(
    num_clients: int = 10,
    iid: Optional[bool] = True,
    balance: Optional[bool] = True,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid or non iid partitions to simulate the federated.

    setting.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    iid : bool, optional
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed by
        chunks to each client (used to test the convergence in a worst case scenario),
        by default True
    balance : bool, optional
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the
        model.
    """
    trainset, testset = _download_data()
    partition_size = int(len(trainset) / num_clients)
    lengths = [partition_size] * num_clients
    if iid:
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
    else:
        if balance:
            trainset = _balance_classes(trainset, seed)
            partition_size = int(len(trainset) / num_clients)
        shard_size = int(partition_size / 2)
        idxs = trainset.targets.argsort()
        sorted_data = Subset(trainset, idxs)
        tmp = []
        for idx in range(num_clients * 2):
            tmp.append(
                Subset(sorted_data, np.arange(shard_size * idx, shard_size * (idx + 1)))
            )
        idxs_list = torch.randperm(
            num_clients * 2, generator=torch.Generator().manual_seed(seed)
        )
        datasets = [
            ConcatDataset((tmp[idxs_list[2 * i]], tmp[idxs_list[2 * i + 1]]))
            for i in range(num_clients)
        ]

    return datasets, testset


def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    Dataset
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in class_counts:
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(trainset.targets[idxs[int(count) : int(count + smallest)]])
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled
