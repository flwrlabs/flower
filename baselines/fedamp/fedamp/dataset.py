"""FedAMP: Personalized Cross-Silo Federated Learning on Non-IID Data."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

FDS = None  # Cache FederatedDataset


def load_data(
    dataset: str,
    num_classes_per_partition: int,
    partition_by: str,
    partition_id: int,
    num_partitions: int,
    batch_size=32,
):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            num_classes_per_partition=num_classes_per_partition,
            partition_by=partition_by,
            class_assignment_mode="first-deterministic",
        )
        if dataset == "cifar10":
            FDS = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )

        elif dataset == "fashion":
            FDS = FederatedDataset(
                dataset="zalando-datasets/fashion_mnist",
                partitioners={"train": partitioner},
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    if dataset == "cifar10":
        pytorch_transforms = Compose(
            [
                ToTensor(),
                Normalize(
                    (0.49139968, 0.48215827, 0.44653124),
                    (0.24703233, 0.24348505, 0.26158768),
                ),
            ]  # mean and std for cifar-10
        )
    elif dataset == "fashion":
        pytorch_transforms = Compose(
            [
                ToTensor(),
                Normalize(
                    (0.286,),
                    (0.353,),
                ),
            ]  # mean and std for fashion mnsit
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if FDS is not None:
        partition = FDS.load_partition(partition_id)
    # rename image column for fashiion dataset
    if dataset == "fashion":
        partition = partition.rename_column("image", "img")

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader
