"""fedpac: A Flower Baseline."""


from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split, DataLoader
from torchvision.datasets import CIFAR10, EMNIST
import torchvision.transforms as transforms

def get_label_list(dataset: Dataset) -> Dict[int, int]:
    """Extract label distribution from a dataset."""
    if hasattr(dataset, "targets"):
        # Convert targets to numpy array safely
        if isinstance(dataset.targets, torch.Tensor):
            targets = dataset.targets.numpy()
        else:
            targets = np.asarray(dataset.targets, dtype=np.int64)
        labels, counts = np.unique(targets, return_counts=True)
        return dict(zip(labels, counts))

    # Fallback for datasets without .targets
    label_counter = {}
    for _, label in dataset:
        if isinstance(label, torch.Tensor):
            label = label.item()
        label_counter[label] = label_counter.get(label, 0) + 1
    return label_counter

def _balance_classes(trainset: Dataset, seed: Optional[int] = 42) -> Dataset:
    """Balance classes in dataset."""
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

def load_data(
    partition_id: int,
    num_partitions: int,
    iid: bool = False,
    balance: bool = True,
    dataset: str = "cifar10",
    s: float = 0.2,
    sample_size: int = 600,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Load partitioned data."""
    # Define transforms
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform_train)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform_test)
        num_classes = 10
        noniid_labels_list = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0]]
        client_group = partition_id % 5
    elif dataset == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = EMNIST("./dataset", split="byclass", train=True, download=True, transform=transform)
        testset = EMNIST("./dataset", split="byclass", train=False, download=True, transform=transform)
        num_classes = 62
        noniid_labels_list = [list(range(10)), list(range(10, 36)), list(range(36, 62))]
        client_group = partition_id % 3
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if balance:
        trainset = _balance_classes(trainset, seed)

    if iid:
        total_len = len(trainset)
        lengths = [total_len // num_partitions] * num_partitions
        remainder = total_len % num_partitions
        for i in range(remainder):
            lengths[i] += 1
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
        train_data = datasets[partition_id]
    else:
        train_data = _create_noniid_partition(
            trainset,
            noniid_labels_list[client_group],
            s,
            sample_size,
            seed,
            num_classes
        )

    # Create data loaders
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    
    return trainloader, testloader

def _create_noniid_partition(
    dataset: Dataset,
    noniid_labels: List[int],
    s: float,
    sample_size: int,
    seed: int,
    num_classes: int
) -> Dataset:
    """Create non-IID partition for a client."""
    # Convert targets to numpy array safely
    if isinstance(dataset.targets, torch.Tensor):
        targets = dataset.targets.numpy()
    else:
        targets = np.asarray(dataset.targets, dtype=np.int64)
    
    # Get indices for each class
    all_indices = {label: np.where(targets == label)[0] for label in range(num_classes)}

    # Calculate partition sizes
    iid_partition_size = int(sample_size * s)
    noniid_partition_size = sample_size - iid_partition_size
    
    # Use a seeded random number generator for reproducibility
    rng = np.random.default_rng(seed)
    selected_indices = []
    
    # Add IID data - small portion from each class
    iid_samples_per_class = [iid_partition_size // num_classes] * num_classes
    remainder = iid_partition_size % num_classes
    for i in range(remainder):
        iid_samples_per_class[i] += 1
    
    for label, count in enumerate(iid_samples_per_class):
        if count > 0:
            selected = rng.choice(all_indices[label], count, replace=False)
            selected_indices.extend(selected)
    
    # Add non-IID data - larger portions from specified classes
    num_dominant_classes = len(noniid_labels)
    noniid_samples_per_class = [noniid_partition_size // num_dominant_classes] * num_dominant_classes
    remainder = noniid_partition_size % num_dominant_classes
    for i in range(remainder):
        noniid_samples_per_class[i] += 1

    for i, label in enumerate(noniid_labels):
        count = noniid_samples_per_class[i]
        if count > 0:
            # Ensure we don't re-select indices from the IID part
            available_indices = np.setdiff1d(all_indices[label], selected_indices)
            # Handle cases where there are not enough samples
            num_to_select = min(count, len(available_indices))
            selected = rng.choice(available_indices, num_to_select, replace=False)
            selected_indices.extend(selected)
               
    # Create final dataset
    partition = Subset(dataset, selected_indices)
    
    return partition