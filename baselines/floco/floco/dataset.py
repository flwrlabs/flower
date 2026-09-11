"""floco: A Flower Baseline."""

from flwr.app import Context
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import InnerDirichletPartitioner
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_dataset
from .partitioners import FoldPartitioner

# pylint: disable=C0103, W0603

# Cache datasets and dataloaders
global_test_set = None
fds = None


def get_testloader(dataset: str) -> DataLoader:
    """Create the global test DataLoader for a specified dataset."""
    global global_test_set
    if global_test_set is None:
        if dataset == "CIFAR10":
            global_test_set = load_dataset("uoft-cs/cifar10", split="test")
        else:
            raise NotImplementedError("Dataset not implemented")
    testloader = DataLoader(
        global_test_set.with_transform(apply_transforms), batch_size=32
    )
    return testloader


def get_federated_dataloaders(
    partition_id: int, num_partitions: int, context: Context
) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders for a specified dataset and partition.

    partition_id : int
        The ID of the partition to load.
    num_partitions : int
        The total number of partitions.
    dataset : str, optional
        The name of the dataset to load, by default "CIFAR10".
    dataset_split : str, optional
        The method to split the dataset, by default "Dirichlet".
    dataset_split_arg : float, optional
        The argument for the dataset split method, by default 0.5.
        The seed for random number generation, by default 0.
    batch_size : int, optional
        The batch size for the dataloaders, by default 50.

    Tuple[DataLoader, DataLoader, DataLoader]
        A tuple containing the training DataLoader, validation DataLoader,
        and test DataLoader for the specified partition.
    """
    dataset = str(context.run_config["dataset"])
    dataset_split = str(context.run_config["dataset-split"])
    dataset_split_arg = float(context.run_config["dataset-split-arg"])
    seed = int(context.run_config["dataset-split-arg"])
    batch_size = int(context.run_config["batch-size"])
    if dataset_split == "Fold":
        partitioner = FoldPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            num_folds=dataset_split_arg,
            q=20,
            min_partition_size=500,
            seed=seed,
        )
    elif dataset_split == "Dirichlet":
        partitioner = InnerDirichletPartitioner(
            partition_sizes=[500] * num_partitions,
            partition_by="label",
            alpha=dataset_split_arg,
            seed=seed,
        )
    else:
        raise NotImplementedError(f"Unknown dataset split method '{dataset_split}'.")
    # Only initialize `FederatedDataset` and partitions only once
    global fds
    if fds is None:
        fds = FederatedDataset(
            dataset=dataset.lower(), partitioners={"train": partitioner}
        )
    train_partition = fds.load_partition(partition_id, "train")
    train_val_partition = train_partition.train_test_split(test_size=0.2, seed=seed)
    trainloader = DataLoader(
        train_val_partition["train"].with_transform(apply_transforms),
        batch_size=batch_size,
        shuffle=True,
    )
    valloader = DataLoader(
        train_val_partition["test"].with_transform(apply_transforms),
        batch_size=batch_size,
        shuffle=False,
    )
    return trainloader, valloader


def apply_transforms(batch):
    """Apply transforms to a batch."""
    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201]
            ),
        ]
    )
    batch["img"] = [img_transforms(img) for img in batch["img"]]
    return batch
