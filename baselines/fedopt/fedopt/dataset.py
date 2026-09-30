"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

from typing import Tuple

from flwr_datasets import FederatedDataset
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


def get_cifar_10_transforms() -> Tuple[Compose, Compose]:
    """Return transforms for train/test datasets for CIFAR-10."""
    normalize_cifar10 = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = Compose(
        [RandomCrop(24), RandomHorizontalFlip(), ToTensor(), normalize_cifar10]
    )
    test_transform = Compose([CenterCrop(24), ToTensor(), normalize_cifar10])
    return train_transform, test_transform


def _apply_cifar_transform(transforms):
    """Return function that applies a transform to each batch."""

    def _apply_transforms(batch):
        batch["img"] = [transforms(img) for img in batch["img"]]
        return batch

    return _apply_transforms


def get_dataloaders(dataset_cfg: DictConfig):
    """Return dataloaders: one per client, and one for global eval on the server."""
    # Generate partitioned dataset using flwr-datasets
    # Via the config we specify the type of partitioner to use as well
    # as the number of partitions
    partitioner = instantiate(dataset_cfg.partitioners)
    p_key = dataset_cfg.partition_key
    fds: FederatedDataset = instantiate(
        dataset_cfg.federated_dataset, partitioners={p_key: partitioner}
    )
    train_tt, test_tt = get_cifar_10_transforms()
    train_loaders = []

    for i in range(dataset_cfg.num_clients):
        partition = fds.load_partition(i, "train").with_transform(
            _apply_cifar_transform(train_tt)
        )
        loader = DataLoader(partition, batch_size=dataset_cfg.batch_size, shuffle=True)
        train_loaders.append(loader)

    # Load the whole test set to be used for centralised evaluation
    test_dataset = fds.load_full("test").with_transform(_apply_cifar_transform(test_tt))
    test_loader = DataLoader(test_dataset, batch_size=dataset_cfg.batch_size_evaluate)

    return train_loaders, test_loader
