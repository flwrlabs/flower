"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms


def shuffle_list(data):
    """This function returns the shuffled data."""
    for i in range(len(data)):
        tmp_len = len(data[i][0])
        index = list(range(tmp_len))
        random.shuffle(index)
        data[i][0], data[i][1] = shuffle_list_data(data[i][0], data[i][1])
    return data


def shuffle_list_data(x, y):
    """This function is a helper function, shuffles an array while maintaining the
    mapping between x and y.
    """
    indices = list(range(len(x)))
    random.shuffle(indices)
    return x[indices], y[indices]


def get_cifar100(iid=False, transform=None, data_path=None):
    """Return CIFAR10 train/test data and labels as numpy arrays."""
    if data_path is None:
        data_path = (
            Path(__file__).parent.parent.parent.resolve().joinpath("data/cifar10")
        )

    data_train = torchvision.datasets.CIFAR100(
        data_path, train=True, download=True, transform=transform
    )
    data_test = torchvision.datasets.CIFAR100(
        data_path, train=False, download=True, transform=transform
    )

    if iid:
        return data_train, data_test

    x_train, y_train = data_train.data.transpose((0, 1, 2, 3)), np.array(
        data_train.targets
    )
    x_test, y_test = data_test.data.transpose((0, 1, 2, 3)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def get_cifar10(iid=False, transform=None, data_path=None):
    """Return CIFAR10 train/test data and labels as numpy arrays."""
    if data_path is None:
        data_path = (
            Path(__file__).parent.parent.parent.resolve().joinpath("data/cifar10")
        )
    data_train = torchvision.datasets.CIFAR10(
        data_path, train=True, download=True, transform=transform
    )
    data_test = torchvision.datasets.CIFAR10(
        data_path, train=False, download=True, transform=transform
    )

    if iid:
        return data_train, data_test

    x_train, y_train = data_train.data.transpose((0, 1, 2, 3)), np.array(
        data_train.targets
    )
    x_test, y_test = data_test.data.transpose((0, 1, 2, 3)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def get_emnist(iid=False, transform=None, data_path=None):
    """Return global train and test datasets for EMNIST."""
    if data_path is None:
        data_path = (
            Path(__file__).parent.parent.parent.resolve().joinpath("data/emnist")
        )
    train_dataset = torchvision.datasets.EMNIST(
        root=str(data_path),
        train=True,
        download=True,
        transform=transform,
        split="balanced",
    )

    test_dataset = torchvision.datasets.EMNIST(
        root=str(data_path),
        train=False,
        download=True,
        transform=transform,
        split="balanced",
    )
    if iid:
        return train_dataset, test_dataset

    x_train, y_train = train_dataset.data.cpu().numpy().transpose((0, 1, 2)), np.array(
        train_dataset.targets
    )
    x_test, y_test = test_dataset.data.cpu().numpy().transpose((0, 1, 2)), np.array(
        test_dataset.targets
    )

    return x_train, y_train, x_test, y_test


def get_mnist(iid=False, transform=None, data_path=None):
    """Return global train and test datasets for MNIST."""
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent.resolve().joinpath("data/mnist")

    train_dataset = torchvision.datasets.MNIST(
        root=str(data_path), train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=str(data_path), train=False, download=True, transform=transform
    )
    if iid:
        return train_dataset, test_dataset

    x_train, y_train = train_dataset.data.cpu().numpy().transpose((0, 1, 2)), np.array(
        train_dataset.targets
    )
    x_test, y_test = test_dataset.data.cpu().numpy().transpose((0, 1, 2)), np.array(
        test_dataset.targets
    )

    return x_train, y_train, x_test, y_test


def clients_rand(train_len, n_clients):
    """This function creates a random distribution for the local datasets' size, i.e.
    number of images each client possess.
    """
    client_tmp = np.random.randint(10, 100, n_clients)
    sum_ = np.sum(client_tmp)
    clients_dist = (np.floor((client_tmp / sum_) * train_len)).astype(int)
    to_ret = list(clients_dist)
    to_ret[-1] += train_len - clients_dist.sum()
    return to_ret


def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True):
    """Splits (data, labels) among 'n_clients s.t.

    every client can holds 'classes_per_client' number of classes
    Input:
      data : [n_data x shape]
      labels : [n_data (x 1)] from 0 to n_labels
      n_clients : number of clients
      classes_per_client : number of classes per client
      shuffle : True/False => True for shuffling the dataset, False otherwise
    Output:
      clients_split : client data into desired format
    """
    data.shape[0]
    n_labels = np.max(labels) + 1

    data_per_client = clients_rand(len(data), n_clients)
    data_per_client_per_class = [
        np.maximum(1, nd // classes_per_client) for nd in data_per_client
    ]

    # sort for labels
    data_idcs = [[] for _ in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []

        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    clients_split = np.array(clients_split)

    return clients_split


class CustomImageDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        # self.inputs = torch.Tensor(inputs)
        self.inputs = inputs
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return self.inputs.shape[0]


def get_default_data_transforms(dataset, verbose=False):
    if "mnist" in dataset:
        transforms_train = {
            "general": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(*[0.1307, 0.3081])]
            )
        }
        transforms_eval = {
            "general": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(*[0.1307, 0.3081])]
            )
        }
    else:
        transforms_train = {
            "general": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5),
                ]
            )
        }
        transforms_eval = {
            "general": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5),
                ]
            )
        }
    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train["general"].transforms:
            print(" -", transformation)
        print()

    return transforms_train["general"], transforms_eval["general"]


def split_dataset(dataset: torch.utils.data.dataset, n_clients, lengths_list=None):
    """Return iid-split datasets."""
    if lengths_list is None:
        int_lengths = int(len(dataset) / n_clients)
        lengths_list = [int_lengths] * (n_clients - 1)
        lengths_list.append(len(dataset) - (int_lengths * (n_clients - 1)))
    return random_split(dataset=dataset, lengths=lengths_list)


@hydra.main(config_path="conf", config_name="base", version_base=None)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Does everything needed to get the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. print parsed config
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    download_and_preprocess()


dataset_dict = {
    "mnist": get_mnist,
    "cifar10": get_cifar10,
    "cifar100": get_cifar100,
    "emnist": get_emnist,
}
