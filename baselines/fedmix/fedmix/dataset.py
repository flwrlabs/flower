"""..."""

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from fedmix.dataset_preparation import (
    _download_cifar10,
    _download_cifar100,
    _download_femnist,
    _mash_data,
    _partition_cifar,
    _partition_cifar_new,
    _partition_cifar_new_new,
    _partition_femnist
)


def load_datasets(config, num_clients, seed):
    """..."""
    print(f"Dataset partition config: {OmegaConf.to_yaml(config)}")

    dataset_name = config.name
    batch_size = config.batch_size

    if dataset_name == "cifar10":
        trainset, testset = _download_cifar10()
        num_classes = 10
    elif dataset_name == "cifar100":
        trainset, testset = _download_cifar100()
        num_classes = 100
    elif dataset_name == 'femnist':
        _download_femnist(num_clients)
        num_classes = 62
    else:
        raise Exception('dataset_name must be one of ["cifar10", "cifar100", "femnist"]')

    if dataset_name in ['cifar10', 'cifar100']:
        client_datasets = _partition_cifar_new_new(trainset, num_classes, num_clients, config.num_classes_per_client, seed)
    else:
        client_datasets, testset = _partition_femnist(num_clients)

    mashed_data = _mash_data(client_datasets, config.mash_batch_size, num_classes)

    trainloaders = []
    for client_dataset in client_datasets:
        trainloaders.append(
            DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        )

    return trainloaders, DataLoader(testset, batch_size=batch_size), mashed_data
