"""Util functions for CIFAR10/100."""

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from flwr.common import ndarrays_to_parameters
from flwr.common.typing import NDArrays, Parameters, Scalar
from flwr.server.history import History
from PIL import Image
from torch import Tensor, load
from torch.nn import GroupNorm, Module
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import ResNet, resnet18
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from flwr_baselines.dataset.utils.common import (
    XY,
    create_lda_partitions,
    shuffle,
    sort_by_label,
    split_array_at_indices,
)

CIFAR100_NUM_COARSE_CLASSES = 20
CIFAR100_NUM_FINE_CLASSES = 5

cifar100_coarse_to_real = [
    [4, 30, 55, 72, 95],
    [1, 32, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 28, 61],
    [0, 51, 53, 57, 83],
    [22, 39, 40, 86, 87],
    [5, 20, 25, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 37, 68, 76],
    [23, 33, 49, 60, 71],
    [15, 19, 21, 31, 38],
    [34, 63, 64, 66, 75],
    [26, 45, 77, 79, 99],
    [2, 11, 35, 46, 98],
    [27, 29, 44, 78, 93],
    [36, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]
# fmt: off
cifar100_real_to_coarse = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]
# fmt: on


# transforms
def get_transforms(num_classes: int = 10) -> Dict[str, Compose]:
    """Returns the right Transform Compose for both train and evaluation.

    Args:
        num_classes (int, optional): Defines whether CIFAR10 or CIFAR100. Defaults to 10.

    Returns:
        Dict[str, Compose]: Dictionary with 'train' and 'test' keywords and Transforms
        for each
    """
    normalize_cifar10 = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normalize_cifar100 = Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    normalize_cifar = normalize_cifar10 if num_classes == 10 else normalize_cifar100
    train_transform = Compose(
        [RandomCrop(24), RandomHorizontalFlip(), ToTensor(), normalize_cifar]
    )
    test_transform = Compose([CenterCrop(24), ToTensor(), normalize_cifar])
    return {"train": train_transform, "test": test_transform}


def get_cifar_model(num_classes: int = 10) -> Module:
    """Generates ResNet18 model using GroupNormalization rather than
    BatchNormalization. Two groups are used.

    Args:
        num_classes (int, optional): Number of classes {10,100}. Defaults to 10.

    Returns:
        Module: ResNet18 network.
    """
    model: ResNet = resnet18(
        norm_layer=lambda x: GroupNorm(2, x), num_classes=num_classes
    )
    return model


class ClientDataset(Dataset):
    """Client Dataset."""

    def __init__(self, path_to_data: Path, transform: Compose = None):
        """Implements local dataset.

        Args:
            path_to_data (Path): Path to local '.pt' file is located.
            transform (Compose, optional): Transforms to be used when sampling.
            Defaults to None.
        """
        super().__init__()
        self.transform = transform
        self.inputs, self.labels = load(path_to_data)

    def __len__(self) -> int:
        """Size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Fetches item in dataset.

        Args:
            idx (int): Position of item being fetched.

        Returns:
            Tuple[Tensor, int]: Tensor image and respective label
        """
        this_input = Image.fromarray(self.inputs[idx])
        this_label = self.labels[idx]
        if self.transform:
            this_input = self.transform(this_input)

        return this_input, this_label


def save_partitions(
    list_partitions: List[XY], fed_dir: Path, partition_type: str = "train"
):
    """Saves partitions to individual files.

    Args:
        list_partitions (List[XY]): List of partitions to be saves
        fed_dir (Path): Root directory where to save partitions.
        partition_type (str, optional): Partition type ("train" or "test"). Defaults to "train".
    """
    for idx, partition in enumerate(list_partitions):
        path_dir = fed_dir / f"{idx}"
        path_dir.mkdir(exist_ok=True, parents=True)
        torch.save(partition, path_dir / f"{partition_type}.pt")


def partition_cifar10_and_save(
    dataset: XY,
    fed_dir: Path,
    dirichlet_dist: Optional[npt.NDArray[np.float32]] = None,
    num_partitions: int = 500,
    concentration: float = 0.1,
) -> np.ndarray:
    """Creates and saves partitions for CIFAR10.

    Args:
        dataset (XY): Original complete dataset.
        fed_dir (Path): Root directory where to save partitions.
        dirichlet_dist (Optional[npt.NDArray[np.float32]], optional):
            Pre-defined distributions to be used for sampling if exist. Defaults to None.
        num_partitions (int, optional): Number of partitions. Defaults to 500.
        concentration (float, optional): Alpha value for Dirichlet. Defaults to 0.1.

    Returns:
        np.ndarray: Generated dirichlet distributions.
    """
    # Create partitions
    clients_partitions, dist = create_lda_partitions(
        dataset=dataset,
        dirichlet_dist=dirichlet_dist,
        num_partitions=num_partitions,
        concentration=concentration,
    )
    # Save partions
    save_partitions(list_partitions=clients_partitions, fed_dir=fed_dir)

    return dist


def gen_cifar10_partitions(
    path_original_dataset: Path,
    dataset_name: str,
    num_total_clients: int,
    lda_concentration: float,
) -> Path:
    """Defines root path for partitions and calls functions to create them.

    Args:
        path_original_dataset (Path): Path to original (unpartitioned) dataset.
        dataset_name (str): Friendly name to dataset.
        num_total_clients (int): Number of clients.
        lda_concentration (float): Concentration (alpha) used when generation Dirichlet
        distributions.

    Returns:
        Path: [description]
    """
    fed_dir = (
        path_original_dataset
        / f"{dataset_name}"
        / "partitions"
        / f"{num_total_clients}"
        / f"{lda_concentration:.2f}"
    )

    trainset = CIFAR10(root=path_original_dataset, train=True, download=True)
    flwr_trainset = (trainset.data, np.array(trainset.targets, dtype=np.int32))
    partition_cifar10_and_save(
        dataset=flwr_trainset,
        fed_dir=fed_dir,
        dirichlet_dist=None,
        num_partitions=num_total_clients,
        concentration=lda_concentration,
    )

    return fed_dir


def shuffle_and_create_cifar100_lda_dists(
    dataset: XY,
    lda_concentration_coarse: float,
    lda_concentration_fine: float,
    num_partitions: int,
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], np.ndarray, np.ndarray]:
    """Shuffles the original dataset and creates the two-level LDA
    distributions.

    Args:
        dataset (XY): original dataset in XY format
        lda_concentration_coarse (float): Concentration for coarse (first) level
        lda_concentration_fine (float): Concentration for coarse (second) level
        num_partitions (int): Number of partitions

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray]: organized list of
        yet-to-be-partioned dataset and LDA distributions.
    """

    x_orig, y_orig = shuffle(dataset[0], np.array(dataset[1], dtype=np.int32))

    x_orig, y_orig = sort_by_label(x_orig, y_orig)

    _, start_idx = np.unique(y_orig, return_index=True)
    x_list = split_array_at_indices(x_orig, start_idx)
    y_list = split_array_at_indices(y_orig, start_idx)

    lda_concentration_coarse_vector = np.repeat(
        lda_concentration_coarse, CIFAR100_NUM_COARSE_CLASSES
    )
    lda_concentration_fine_vector = np.repeat(
        lda_concentration_fine, CIFAR100_NUM_FINE_CLASSES
    )

    coarse_dist = np.random.default_rng().dirichlet(
        alpha=lda_concentration_coarse_vector, size=num_partitions
    )
    fine_dist = np.random.default_rng().dirichlet(
        alpha=lda_concentration_fine_vector,
        size=(num_partitions, CIFAR100_NUM_COARSE_CLASSES),
    )
    return x_list, y_list, coarse_dist, fine_dist


def partition_cifar100_and_save(
    dataset: XY,
    fed_dir: Path,
    num_partitions: int,
    lda_concentration_coarse: float,
    lda_concentration_fine: float,
):
    # pylint: disable-msg=too-many-locals
    """Partitions CIFAR100 and saves local datasets.

    Args:
        dataset (XY): Dataset to be partitioned
        fed_dir (Path): Root directory where to save partitions
        num_partitions (int): Number of partitions
        lda_concentration_coarse (float): Concentration for the higer-level classes
        lda_concentration_fine (float): Concentration for fine labels.
    """

    x_list, y_list, coarse_dist, fine_dist = shuffle_and_create_cifar100_lda_dists(
        dataset, lda_concentration_coarse, lda_concentration_fine, num_partitions
    )

    # Assuming balanced distribution
    len_dataset = len(dataset[1])

    remaining_samples_counter = (len_dataset // 100) * np.ones(
        (CIFAR100_NUM_COARSE_CLASSES, CIFAR100_NUM_FINE_CLASSES)
    )

    partitions = []
    for client_id in range(num_partitions):
        x_this_client, y_this_client = [], []

        for _ in range(len_dataset // num_partitions):
            coarse_class = np.random.choice(
                CIFAR100_NUM_COARSE_CLASSES, p=coarse_dist[client_id]
            )
            fine_class = np.random.choice(
                CIFAR100_NUM_FINE_CLASSES, p=fine_dist[client_id][coarse_class]
            )
            real_class = cifar100_coarse_to_real[coarse_class][fine_class]

            # obtain sample
            sample_x: np.ndarray = x_list[real_class][0]
            x_list[real_class] = np.delete(x_list[real_class], 0, 0)  # type: ignore

            sample_y: np.ndarray = y_list[real_class][0]
            y_list[real_class] = np.delete(y_list[real_class], 0, 0)  # type: ignore

            x_this_client.append(sample_x)
            y_this_client.append(sample_y)

            # Update and renormalize
            # check fine class is empty
            remaining_samples_counter[coarse_class, fine_class] -= 1
            if remaining_samples_counter[coarse_class, fine_class] == 0:
                for k in range(num_partitions):
                    fine_dist[k][coarse_class][fine_class] = 0.0
                    norm_factor = np.sum(fine_dist[k][coarse_class])
                    if norm_factor > 0:
                        fine_dist[k][coarse_class] = (
                            fine_dist[k][coarse_class] / norm_factor
                        )
            # Check coarse class is empty
            if np.sum(remaining_samples_counter[coarse_class]) == 0:
                for k in range(num_partitions):
                    coarse_dist[k][coarse_class] = 0.0
                    norm_factor = np.sum(coarse_dist[k])
                    if norm_factor > 0.0:
                        coarse_dist[k] = coarse_dist[k] / norm_factor

        partitions.append(
            (np.array(x_this_client), np.array(y_this_client, dtype=np.int64))
        )
    save_partitions(list_partitions=partitions, fed_dir=fed_dir, partition_type="train")


def gen_cifar100_partitions(
    path_original_dataset: Path,
    dataset_name: str,
    num_total_clients: int,
    lda_concentration_coarse: float,
    lda_concentration_fine: float,
) -> Path:
    """Generates CIFAR100 partitions and return root directory where the
    partitions are.

    Args:
        path_original_dataset (Path): Path to original dataset
        dataset_name (str): Dataset name
        num_total_clients (int): Number of total clients/partitions
        lda_concentration_coarse (float): Concentration for first level LDA
        lda_concentration_fine (float): Concentration for second level LDA

    Returns:
        Path: Path to where partitions are saved
    """
    fed_dir = (
        path_original_dataset
        / f"{dataset_name}"
        / "partitions"
        / f"{num_total_clients}"
        / f"{lda_concentration_coarse:.2f}_{lda_concentration_fine:.2f}"
    )

    trainset = CIFAR100(root=path_original_dataset, train=True, download=True)
    trainset_xy = (trainset.data, trainset.targets)
    partition_cifar100_and_save(
        dataset=trainset_xy,
        fed_dir=fed_dir,
        num_partitions=num_total_clients,
        lda_concentration_coarse=lda_concentration_coarse,
        lda_concentration_fine=lda_concentration_fine,
    )

    return fed_dir


def train(
    net: Module,
    trainloader: DataLoader,
    epochs: int,
    device: str,
    learning_rate: float = 0.01,
) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net: Module, testloader: DataLoader, device: str) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def gen_on_fit_config_fn(
    epochs_per_round: int, batch_size: int, client_learning_rate: float
) -> Callable[[int], Dict[str, Scalar]]:
    """Generates ` On_fit_config`

    Args:
        epochs_per_round (int):  number of local epochs.
        batch_size (int): Batch size
        client_learning_rate (float): Learning rate of clinet

    Returns:
        Callable[[int], Dict[str, Scalar]]: Function to be called at the beginnig of each rounds.
    """

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with specific client learning rate."""
        local_config: Dict[str, Scalar] = {
            "epoch_global": server_round,
            "epochs": epochs_per_round,
            "batch_size": batch_size,
            "client_learning_rate": client_learning_rate,
        }
        return local_config

    return on_fit_config


def get_cifar_eval_fn(
    path_original_dataset: Path, num_classes: int = 10
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Returns an evaluation function for centralized evaluation."""
    CIFAR = CIFAR10 if num_classes == 10 else CIFAR100
    transforms = get_transforms(num_classes=num_classes)

    testset = CIFAR(
        root=path_original_dataset,
        train=False,
        download=True,
        transform=transforms["test"],
    )

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire CIFAR-10 test set for evaluation."""
        # determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = get_cifar_model(num_classes=num_classes)
        state_dict = OrderedDict(
            {
                k: torch.tensor(np.atleast_1d(v))
                for k, v in zip(net.state_dict().keys(), parameters_ndarrays)
            }
        )
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


def get_initial_parameters(num_classes: int = 10) -> Parameters:
    """Returns initial parameters from a model.

    Args:
        num_classes (int, optional): Defines if using CIFAR10 or 100. Defaults to 10.

    Returns:
        Parameters: Parameters to be sent back to the server.
    """
    model = get_cifar_model(num_classes)
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(weights)

    return parameters


def plot_metric_from_history(
    hist: History,
    dataset_name: str,
    strategy_name: str,
    expected_maximum: float,
    save_plot_path: Path,
) -> None:
    """Simple plotting method for Classification Task.

    Args:
        hist (History): Object containing evaluation for all rounds.
        dataset_name (str): Name of the dataset.
        strategy_name (str): Strategy being used
        expected_maximum (float): Expected final accuracy.
        save_plot_path (Path): Where to save the plot.
    """
    rounds, values = zip(*hist.metrics_centralized["accuracy"])
    plt.figure()
    plt.plot(rounds, np.asarray(values) * 100, label=strategy_name)  # Accuracy 0-100%
    # Set expected graph
    plt.axhline(y=expected_maximum, color="r", linestyle="--")
    plt.title(f"Centralized Validation - {dataset_name}")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(save_plot_path)
    plt.close()
