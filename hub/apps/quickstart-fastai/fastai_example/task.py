"""fastai_example: A Flower / Fastai app."""

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Resize, ToTensor

fds = None  # Cache FederatedDataset


def load_data(
    partition_id,
    num_partitions,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
            trust_remote_code=True,
        )
    partition = fds.load_partition(partition_id, "train")

    # Resize and repeat channels to use MNIST, which have grayscale images,
    # with squeezenet, which expects 3 channels.
    # Ref: https://discuss.pytorch.org/t/fine-tuning-squeezenet-for-mnist-dataset/31221/2
    pytorch_transforms = Compose(
        [Resize(224), ToTensor(), Lambda(lambda x: x.expand(3, -1, -1))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    def collate_fn(batch):
        """Change the dictionary to tuple to keep the exact dataloader behavior."""
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)

        return images_tensor, labels_tensor

    partition = partition.with_transform(apply_transforms)
    # 20 % for on federated evaluation
    partition_full = partition.train_test_split(test_size=0.2, seed=42)
    # 60 % for the federated train and 20 % for the federated validation (both in fit)
    partition_train_valid = partition_full["train"].train_test_split(
        train_size=0.75, seed=42
    )
    trainloader = DataLoader(
        partition_train_valid["train"],
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valloader = DataLoader(
        partition_train_valid["test"],
        batch_size=32,
        collate_fn=collate_fn,
    )
    testloader = DataLoader(
        partition_full["test"], batch_size=32, collate_fn=collate_fn, num_workers=1
    )
    return trainloader, valloader, testloader
