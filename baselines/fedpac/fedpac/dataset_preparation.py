import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, EMNIST


def get_label_list(dataset):
    label_counter = {}
    for _, labels in dataset:
        if isinstance(labels, int) or isinstance(labels, torch.tensor) :
            label_counter[int(labels)] = label_counter.get(int(labels), 0) + 1
        else:
            for label in labels:
                label_counter[int(label)] = label_counter.get(int(label), 0) + 1
    return label_counter


def _download_data(dataset: str) -> Tuple[Dataset, Dataset]:
    """Download (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    if dataset == "emnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        trainset = EMNIST(
            "./dataset", split="byclass", train=True, download=True, transform=transform
        )
        testset = EMNIST(
            "./dataset",
            split="byclass",
            train=False,
            download=True,
            transform=transform,
        )

    elif dataset == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        trainset = CIFAR10(
            "./dataset", train=True, download=True, transform=transform_train
        )
        testset = CIFAR10(
            "./dataset", train=False, download=True, transform=transform_test
        )

    return trainset, testset


def partition_cifar_data(
    trainset,
    num_clients,
    iid: Optional[bool] = False,
    balance: Optional[bool] = True,
    s: Optional[float] = 0.2,
    sample_size: int = 600,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid or non iid partitions to simulate the.

        federated setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    iid : bool, optional
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default False
    s : float, optional
        fraction of iid data for each client. 0.2 by default from paper.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the model.
    """
    if balance:
        trainset = _balance_classes(trainset, seed)

    partition_size = int(len(trainset) / num_clients)
    # sample_size=partition_size
    lengths = [partition_size] * num_clients
    labels = np.unique(trainset.targets)
    num_classes = len(labels)

    if iid:
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
        return datasets
    else:
        noniid_labels_list = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0]]
        iid_partition_size = int(sample_size * s)
        noniid_partition_size = sample_size - iid_partition_size
        idxs = np.array(trainset.targets).argsort()
        iid_shard_size = int(iid_partition_size / num_classes)
        sorted_data = Subset(trainset, idxs)

        label_count = get_label_list(sorted_data)
        label_index = {0: 0}
        cumulative_sum = 0
        for key in label_count.keys():
            cumulative_sum += label_count[key]
            label_index[key + 1] = cumulative_sum
        client_data = {}
        li = copy.copy(label_index)
        for idx in range(num_clients):
            noniid_shard_size = int(
                noniid_partition_size / len(noniid_labels_list[idx % 5])
            )
            client_data[idx] = []
            for i in labels:  # Partition IID Data
                start = label_index[i]
                stop = label_index[i] + iid_shard_size
                if stop > li[i + 1]:
                    start = li[i]
                    stop = start + iid_shard_size
                    if stop > li[i + 1]:
                        start = li[i]
                        stop = li[i + 1] - 1
                    label_index[i] = li[i]
                else:
                    label_index[i] += iid_shard_size

                data_array = np.arange(start, stop)
                client_data[idx].append(Subset(sorted_data, data_array))

            noniid_labels = noniid_labels_list[idx % 5]  # Partition Non-IID Data
            for ni in noniid_labels:
                start = label_index[ni]
                stop = label_index[ni] + noniid_shard_size
                if stop > label_index[ni + 1]:
                    start = label_index[ni]
                    stop = start + noniid_shard_size
                    if stop > label_index[ni + 1]:
                        start = label_index[ni]
                        stop = label_index[ni + 1] - 1
                    label_index[ni] = label_index[ni]

                else:
                    label_index[ni] += noniid_shard_size

                data_array = np.arange(start, stop)
                client_data[idx].append(Subset(sorted_data, data_array))
                label_index[ni] += noniid_shard_size
        datasets = [ConcatDataset(client_data[c]) for c in range(num_clients)]
        l = get_label_list(datasets[-1])
        print(l)
    return datasets


def partition_emnist_data(
    trainset,
    num_clients,
    iid: Optional[bool] = False,
    balance: Optional[bool] = True,
    s: Optional[float] = 0.2,
    sample_size: int = 1000,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid or non iid partitions to simulate.

    the federated setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    iid : bool, optional
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default False
    s : float, optional
        fraction of iid data for each client. 0.2 by default from paper.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the model.
    """

    if balance:
        trainset = _balance_classes(trainset, seed)

    partition_size = int(len(trainset) / num_clients)
    # sample_size = partition_size
    lengths = [partition_size] * num_clients
    labels = np.unique(trainset.targets)
    num_classes = len(labels)

    if iid:
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
        return datasets

    else:
        noniid_labels_list = [
            list(range(10)),
            list(range(10, 36)),
            list(range(36, 62)),
        ]
        iid_partition_size = int(sample_size * s)
        noniid_partition_size = sample_size - iid_partition_size
        idxs = trainset.targets.argsort()
        iid_shard_size = int(iid_partition_size / num_classes)
        sorted_data = Subset(trainset, idxs)

        label_count = get_label_list(sorted_data)
        label_index = {0: 0}
        cumulative_sum = 0
        for key in label_count.keys():
            cumulative_sum += label_count[key]
            label_index[key + 1] = cumulative_sum
        client_data = {}
        li = copy.copy(label_index)
        for idx in range(num_clients):
            noniid_shard_size = int(
                noniid_partition_size / len(noniid_labels_list[idx % 3])
            )
            client_data[idx] = []
            for i in labels:  # Partition IID Data
                start = label_index[i]
                stop = label_index[i] + iid_shard_size
                if stop > li[i + 1]:
                    start = li[i]
                    stop = start + iid_shard_size
                    if stop > li[i + 1]:
                        start = li[i]
                        stop = li[i + 1] - 1
                    label_index[i] = li[i]
                else:
                    label_index[i] += iid_shard_size

                data_array = np.arange(start, stop)
                client_data[idx].append(Subset(sorted_data, data_array))

            noniid_labels = noniid_labels_list[idx % 3]  # Partition Non-IID Data
            for ni in noniid_labels:
                start = label_index[ni]
                stop = label_index[ni] + noniid_shard_size
                if stop > label_index[ni + 1]:
                    start = label_index[ni]
                    stop = start + noniid_shard_size
                    if stop > label_index[ni + 1]:
                        start = label_index[ni]
                        stop = label_index[ni + 1] - 1
                    label_index[ni] = label_index[ni]

                else:
                    label_index[ni] += noniid_shard_size

                data_array = np.arange(start, stop)
                client_data[idx].append(Subset(sorted_data, data_array))
                label_index[ni] += noniid_shard_size
    datasets = [ConcatDataset(client_data[c]) for c in range(num_clients)]
    l = get_label_list(datasets[-1])
    print(l)
    return datasets


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
    if isinstance(trainset.targets, list):
        trainset.targets = torch.tensor(trainset.targets)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in np.cumsum(class_counts):
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
