"""fedagain: Data loading, partitioning and client corruption.

Each client receives a partition of the training split (IID, Dirichlet or
label-skew). Corrupted clients are chosen deterministically from the seed. For a
corrupted client, the corruption is applied to its *whole* partition before it is
divided into local training and validation sets, so that the local validation
data shares the degradation of the local training data. This is the assumption
behind the benchmark error E_k (a hospital with a degraded endoscope also
validates on degraded images).
"""

from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, Partitioner
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import pil_to_tensor

from fedagain.corruptions import apply_corruption, parse_corruption_types


@dataclass(frozen=True)
class DatasetSpec:
    """Static information about a supported dataset."""

    hf_name: str
    image_key: str
    label_key: str
    num_classes: int
    in_channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASETS = {
    "cifar10": DatasetSpec(
        hf_name="uoft-cs/cifar10",
        image_key="img",
        label_key="label",
        num_classes=10,
        in_channels=3,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    "mnist": DatasetSpec(
        hf_name="ylecun/mnist",
        image_key="image",
        label_key="label",
        num_classes=10,
        in_channels=1,
        mean=(0.1307,),
        std=(0.3081,),
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` of a supported dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    return DATASETS[name]


class LabelSkewPartitioner(Partitioner):
    """Label-skew partitioner of the paper (Sec. 5.2.2, Eq. 13).

    Client ``k`` has ``num_primary`` primary classes
    ``{(k * num_primary + j) mod C}``. A share ``primary_share`` of the samples of
    class ``c`` is split evenly across the clients for which ``c`` is primary and
    the remaining samples are split evenly across all other clients.
    """

    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        num_primary: int = 2,
        primary_share: float = 0.8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if num_partitions < 2:
            raise ValueError("label-skew partitioning needs at least 2 clients")
        if not 0.0 < primary_share <= 1.0:
            raise ValueError("primary_share must be in (0, 1]")
        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._num_primary = num_primary
        self._primary_share = primary_share
        self._seed = seed
        self._partition_indices: list[list[int]] | None = None

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        return self._num_partitions

    def primary_classes(self, partition_id: int, num_classes: int) -> list[int]:
        """Return the primary classes of a client."""
        return sorted(
            {
                (partition_id * self._num_primary + j) % num_classes
                for j in range(self._num_primary)
            }
        )

    def _compute_partitions(self) -> list[list[int]]:
        labels = np.asarray(self.dataset[self._partition_by])
        classes = np.unique(labels)
        num_classes = int(classes.max()) + 1
        rng = np.random.default_rng(self._seed)
        indices: list[list[int]] = [[] for _ in range(self._num_partitions)]
        for c in classes:
            class_idx = np.flatnonzero(labels == c)
            rng.shuffle(class_idx)
            primary = [
                k
                for k in range(self._num_partitions)
                if c in self.primary_classes(k, num_classes)
            ]
            others = [k for k in range(self._num_partitions) if k not in primary]
            if not primary or not others:
                # Degenerate cases: split class c uniformly across all clients.
                for k, chunk in enumerate(
                    np.array_split(class_idx, self._num_partitions)
                ):
                    indices[k].extend(chunk.tolist())
                continue
            n_primary = int(round(self._primary_share * len(class_idx)))
            for k, chunk in zip(
                primary,
                np.array_split(class_idx[:n_primary], len(primary)),
                strict=False,
            ):
                indices[k].extend(chunk.tolist())
            for k, chunk in zip(
                others, np.array_split(class_idx[n_primary:], len(others)), strict=False
            ):
                indices[k].extend(chunk.tolist())
        return [sorted(idx) for idx in indices]

    def load_partition(self, partition_id: int) -> Dataset:
        """Load a single partition based on the partition index."""
        if self._partition_indices is None:
            self._partition_indices = self._compute_partitions()
        return self.dataset.select(self._partition_indices[partition_id])


def build_partitioner(
    partitioning: str,
    num_partitions: int,
    label_key: str,
    seed: int,
    dirichlet_alpha: float = 0.5,
    label_skew_classes: int = 2,
    label_skew_primary_share: float = 0.8,
) -> Partitioner:
    """Create the Flower Datasets partitioner selected in the run config."""
    if partitioning == "iid":
        return IidPartitioner(num_partitions=num_partitions)
    if partitioning == "dirichlet":
        return DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=label_key,
            alpha=dirichlet_alpha,
            min_partition_size=10,
            self_balancing=False,
            seed=seed,
        )
    if partitioning == "label-skew":
        return LabelSkewPartitioner(
            num_partitions=num_partitions,
            partition_by=label_key,
            num_primary=label_skew_classes,
            primary_share=label_skew_primary_share,
            seed=seed,
        )
    raise ValueError(
        f"Unknown partitioning '{partitioning}'. "
        "Use 'iid', 'dirichlet' or 'label-skew'."
    )


def corrupted_clients(num_partitions: int, fraction: float, seed: int) -> list[int]:
    """Return the (sorted) ids of the clients whose data is corrupted."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"corrupted-fraction must be in [0, 1], got {fraction}")
    num_corrupted = int(round(fraction * num_partitions))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_partitions)
    return sorted(int(k) for k in perm[:num_corrupted])


def is_corrupted_client(
    partition_id: int, num_partitions: int, run_config: dict
) -> bool:
    """Return ``True`` if the data of this client is corrupted."""
    bad = corrupted_clients(
        num_partitions,
        float(run_config["corrupted-fraction"]),
        int(run_config["seed"]),
    )
    return partition_id in bad


def corrupt_images(
    images: torch.Tensor, run_config: dict, gen: torch.Generator
) -> torch.Tensor:
    """Corrupt a share ``corruption-ratio`` of ``images`` (in place).

    Following the paper (Sec. 5.3), each selected image receives one corruption
    drawn at random from ``corruption-types``, at ``corruption-severity``.
    """
    names = parse_corruption_types(str(run_config["corruption-types"]))
    severity = int(run_config["corruption-severity"])
    num_bad = int(round(float(run_config["corruption-ratio"]) * len(images)))
    bad_idx = torch.randperm(len(images), generator=gen)[:num_bad]
    choices = torch.randint(len(names), (num_bad,), generator=gen)
    for i, choice in zip(bad_idx.tolist(), choices.tolist(), strict=True):
        images[i] = apply_corruption(images[i], names[choice], severity, gen)
    return images


def _to_tensors(dataset: Dataset, spec: DatasetSpec) -> tuple[torch.Tensor, ...]:
    """Convert a HF dataset of PIL images into ``[0, 1]`` image and label tensors."""
    images = torch.stack(
        [pil_to_tensor(img) for img in dataset[spec.image_key]]
    ).float()
    images = images / 255.0
    labels = torch.as_tensor(dataset[spec.label_key], dtype=torch.long)
    return images, labels


def _normalize(images: torch.Tensor, spec: DatasetSpec) -> torch.Tensor:
    mean = torch.tensor(spec.mean).view(1, -1, 1, 1)
    std = torch.tensor(spec.std).view(1, -1, 1, 1)
    return (images - mean) / std


FDS: dict[tuple, FederatedDataset] = {}  # Cache of FederatedDataset objects
CLIENT_DATA: dict[tuple, tuple[TensorDataset, TensorDataset, bool]] = {}


def _federated_dataset(run_config: dict, num_partitions: int) -> FederatedDataset:
    spec = get_dataset_spec(str(run_config["dataset"]))
    key = (
        spec.hf_name,
        num_partitions,
        str(run_config["partitioning"]),
        float(run_config["dirichlet-alpha"]),
        int(run_config["label-skew-classes"]),
        float(run_config["label-skew-primary-share"]),
        int(run_config["seed"]),
    )
    if key not in FDS:
        partitioner = build_partitioner(
            partitioning=str(run_config["partitioning"]),
            num_partitions=num_partitions,
            label_key=spec.label_key,
            seed=int(run_config["seed"]),
            dirichlet_alpha=float(run_config["dirichlet-alpha"]),
            label_skew_classes=int(run_config["label-skew-classes"]),
            label_skew_primary_share=float(run_config["label-skew-primary-share"]),
        )
        FDS[key] = FederatedDataset(
            dataset=spec.hf_name, partitioners={"train": partitioner}
        )
    return FDS[key]


def load_client_datasets(
    partition_id: int, num_partitions: int, run_config: dict
) -> tuple[TensorDataset, TensorDataset, bool]:
    """Load the local train/validation sets of a client.

    Returns the training set, the validation set and whether the client is one
    of the corrupted clients. The result is cached per process, so the
    partition is converted and corrupted only once per run.
    """
    key = (partition_id, num_partitions, tuple(sorted(run_config.items())))
    if key in CLIENT_DATA:
        return CLIENT_DATA[key]

    spec = get_dataset_spec(str(run_config["dataset"]))
    seed = int(run_config["seed"])
    fds = _federated_dataset(run_config, num_partitions)
    images, labels = _to_tensors(fds.load_partition(partition_id), spec)

    # Corrupt in [0, 1] pixel space, *before* normalization and before the
    # train/validation split, so the local validation set is degraded as well.
    corrupted = is_corrupted_client(partition_id, num_partitions, run_config)
    if corrupted:
        gen = torch.Generator().manual_seed(seed * 100_003 + partition_id)
        images = corrupt_images(images, run_config, gen)
    images = _normalize(images, spec)

    # Local train/validation split (after corruption, see module docstring)
    gen = torch.Generator().manual_seed(seed + partition_id)
    perm = torch.randperm(len(images), generator=gen)
    num_val = max(1, int(round(float(run_config["val-ratio"]) * len(images))))
    val_idx, train_idx = perm[:num_val], perm[num_val:]
    trainset = TensorDataset(images[train_idx], labels[train_idx])
    valset = TensorDataset(images[val_idx], labels[val_idx])

    CLIENT_DATA[key] = (trainset, valset, corrupted)
    return CLIENT_DATA[key]


def load_client_dataloaders(
    partition_id: int, num_partitions: int, run_config: dict, server_round: int = 0
) -> tuple[DataLoader, DataLoader, bool]:
    """Return train and validation dataloaders of a client.

    The shuffling of the training loader is seeded with the run seed, the client id
    and the server round, so runs are reproducible but batches differ per round.
    """
    trainset, valset, corrupted = load_client_datasets(
        partition_id, num_partitions, run_config
    )
    batch_size = int(run_config["batch-size"])
    gen = torch.Generator().manual_seed(
        int(run_config["seed"]) * 1_000_003 + 1_000 * server_round + partition_id
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, generator=gen
    )
    valloader = DataLoader(valset, batch_size=256)
    return trainloader, valloader, corrupted


def load_centralized_testset(run_config: dict) -> TensorDataset:
    """Load the (clean) test split used for centralized evaluation."""
    spec = get_dataset_spec(str(run_config["dataset"]))
    fds = _federated_dataset(run_config, num_partitions=2)
    images, labels = _to_tensors(fds.load_split("test"), spec)
    return TensorDataset(_normalize(images, spec), labels)
